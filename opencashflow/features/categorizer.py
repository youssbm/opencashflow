"""Transaction categorisation for open-banking and bank-statement data.

The default engine is a transparent, priority-ordered rule set covering common European bank
narratives in seven languages (English, German, French, Polish, Italian, Dutch and Spanish). A
lightweight character n-gram classifier can be trained from labelled data and used as a fallback
when no rule matches.

Design goals:

* **Auditable** - every prediction carries the rule or model that produced it.
* **Multilingual** - text is accent-folded and matched against patterns in seven languages
  (English, German, French, Polish, Italian, Dutch and Spanish).
* **Direction-aware** - the same token ("TRANSFER", "INVOICE") means different things on credits
  and debits, so rules declare the direction they apply to.
* **Extensible** - pass your own :class:`CategoryRule` list; the defaults are only a starting point.

Example:
    >>> from opencashflow.features.categorizer import TransactionCategorizer
    >>> cat = TransactionCategorizer()
    >>> match = cat.categorize("SEPA CREDIT TRANSFER FROM ACME GMBH INVOICE 2024-001", "inflow")
    >>> match.category
    'revenue'
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Direction = Literal["inflow", "outflow", "any"]

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "revenue": "Customer receipts, invoices and sales settlements",
    "other_inflow": "Inflows that could not be assigned to a specific category",
    "transfer_in": "Incoming transfers, often internal or owner funding",
    "loan_proceeds": "Drawdown of a loan or credit facility",
    "equity_injection": "Owner or shareholder equity received",
    "refund": "Refunds and returned funds received",
    "interest_income": "Interest credited by the bank",
    "payroll": "Salaries, wages and social contributions paid to employees",
    "rent": "Rent, lease and property costs",
    "utilities": "Energy, water, telecom and internet",
    "tax": "Corporate income tax and other direct taxes",
    "vat": "VAT, GST and sales-tax payments",
    "supplier": "Supplier, wholesale and purchase invoices",
    "loan_repayment": "Principal and instalment payments on debt",
    "interest_expense": "Interest and financing charges paid",
    "bank_fees": "Account, transaction and card fees",
    "insurance": "Insurance premiums",
    "card_payment": "Card and point-of-sale payments",
    "cash_withdrawal": "ATM and cash withdrawals",
    "transfer_out": "Outgoing transfers, often internal",
    "gambling": "Gambling, betting and lottery",
    "nsf": "Non-sufficient-funds and returned items",
    "other_outflow": "Outflows that could not be assigned to a specific category",
    "uncategorized": "No rule or model matched",
}

INFLOW_CATEGORIES = frozenset(
    {
        "revenue",
        "other_inflow",
        "transfer_in",
        "loan_proceeds",
        "equity_injection",
        "refund",
        "interest_income",
    }
)
OUTFLOW_CATEGORIES = frozenset(
    {
        "payroll",
        "rent",
        "utilities",
        "tax",
        "vat",
        "supplier",
        "loan_repayment",
        "interest_expense",
        "bank_fees",
        "insurance",
        "card_payment",
        "cash_withdrawal",
        "transfer_out",
        "gambling",
        "nsf",
        "other_outflow",
    }
)
REVENUE_CATEGORIES = frozenset({"revenue"})
EXPENSE_CATEGORIES = frozenset(
    {
        "payroll",
        "rent",
        "utilities",
        "tax",
        "vat",
        "supplier",
        "interest_expense",
        "bank_fees",
        "insurance",
        "card_payment",
        "cash_withdrawal",
        "gambling",
        "other_outflow",
    }
)
DEBT_CATEGORIES = frozenset({"loan_repayment", "interest_expense"})

#: Return and rejection vocabulary across the seven supported languages. Match this against
#: :func:`normalize_text` output, which is accent-folded and upper case.
RETURNED_PAYMENT_PATTERN = (
    r"\b(?:RETURN|REJECT|REJET|REJETE|DISHONOUR|DISHONOR|UNPAID|FAILED|BOUNCED|CHARGEBACK|"
    r"RUECKLASTSCHRIFT|RUCKLASTSCHRIFT|NICHT AUSGEFUEHRT|NICHT AUSGEFUHRT|"
    r"NICHT EINGELOEST|EINGELOEST|EINGELOST|"
    r"UNGENUEGEND|UNGENUGEND|ZURUECKGEBUCHT|ZURUCKGEBUCHT|"
    r"IMPAYE|RETOUR|RIFIUT|RESTITUIT|ODRZUC|ZWROT|"
    r"TERUGGEBOEKT|ONVOLDOENDE|GEWEIGERD|DEVUELT|RECHAZ|NO EJECUTAD)"
)

#: Words that identify internal transfers. These are checked before generic revenue/transfer rules.
_INTERNAL_TRANSFER = (
    r"\b(OWN ACCOUNT|INTERNAL TRANSFER|TRANSFER BETWEEN|BETWEEN ACCOUNTS|EIGENES KONTO|"
    r"COMPTE INTERNE|VIREMENT INTERNE|CONTO INTERNO|BONIFICO INTERNO|INTERNE OVERBOEKING|"
    r"PRZELEW WLASNY|UMBUCHUNG|TRANSFERENCIA INTERNA)\b"
)


@dataclass(frozen=True)
class CategoryRule:
    """A single auditable categorisation rule.

    Args:
        category: Target category key, one of :data:`CATEGORY_DESCRIPTIONS`.
        patterns: Regular expressions matched against normalised text. Any match fires the rule.
        direction: ``"inflow"``, ``"outflow"`` or ``"any"``. Direction-aware rules avoid the classic
            mistake of treating a loan drawdown and a loan repayment as the same category.
        priority: Lower numbers run first. Put specific rules before generic ones.
        confidence: Confidence reported when this rule fires.
        description: Human-readable explanation shown in audit output.
    """

    category: str
    patterns: tuple[str, ...]
    direction: Direction = "any"
    priority: int = 100
    confidence: float = 0.9
    description: str = ""
    _compiled: tuple[re.Pattern[str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.category not in CATEGORY_DESCRIPTIONS:
            raise ValueError(
                f"Unknown category {self.category!r}. Add it to CATEGORY_DESCRIPTIONS."
            )
        if self.direction not in {"inflow", "outflow", "any"}:
            raise ValueError(f"Invalid direction {self.direction!r}.")
        patterns = (self.patterns,) if isinstance(self.patterns, str) else tuple(self.patterns)
        if not patterns:
            raise ValueError("A CategoryRule needs at least one pattern.")
        object.__setattr__(self, "patterns", patterns)
        compiled = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
        object.__setattr__(self, "_compiled", compiled)

    def matches(self, text: str, direction: str) -> str | None:
        """Return the matching pattern source, or ``None``.

        ``direction="any"`` means no direction filtering, so a direction-specific rule is still
        consulted in that case.
        """
        if self.direction != "any" and direction != "any" and self.direction != direction:
            return None
        for pattern, regex in zip(self.patterns, self._compiled):
            if regex.search(text):
                return pattern
        return None


def _rule(
    category: str,
    patterns: tuple[str, ...] | str,
    direction: Direction,
    priority: int,
    confidence: float,
    description: str,
) -> CategoryRule:
    if isinstance(patterns, str):
        patterns = (patterns,)
    return CategoryRule(category, patterns, direction, priority, confidence, description)


#: Default rule set. Priorities are intentionally explicit so users can insert rules without
#: rewriting the whole table.
DEFAULT_RULES: tuple[CategoryRule, ...] = (
    _rule(
        "transfer_in",
        (_INTERNAL_TRANSFER,),
        "inflow",
        5,
        0.95,
        "Internal transfer between the borrower's own accounts",
    ),
    _rule(
        "transfer_out",
        (_INTERNAL_TRANSFER,),
        "outflow",
        5,
        0.95,
        "Internal transfer between the borrower's own accounts",
    ),
    _rule(
        "nsf",
        (
            r"\b(NSF|INSUFFICIENT FUNDS|RETURNED (ITEM|PAYMENT|DIRECT DEBIT|CHEQUE)|"
            r"RETURN OF FUNDS|REJECTED|DISHONOU?R|UNPAID|RUECKLASTSCHRIFT|RUCKLASTSCHRIFT|"
            r"LASTSCHRIFT NICHT|ZURUECKGEBUCHT|IMPAYE|REJETE|RIFIUTATO|ODRZUCONY|ODRZUCONA|"
            r"CHARGEBACK|PAYMENT FAILED|DIRECT DEBIT FAILED|TERUGGEBOEKT|DEVUELTO|"
            r"RESTITUZIONE|RECHAZ\w*|GEWEIGERD)\b"
        ),
        "outflow",
        6,
        0.95,
        "Failed or returned payment",
    ),
    _rule(
        "vat",
        (r"\b(VAT|UMSATZSTEUER|UST|TVA|IVA|BTW|GST|SALES TAX|VAT PAYMENT|MOMS|PPN)\b",),
        "outflow",
        8,
        0.95,
        "VAT / sales-tax payment",
    ),
    _rule(
        "tax",
        (
            r"\b(TAX|TAXES|CORPORATION TAX|INCOME TAX|WITHHOLDING TAX|FINANZAMT|STEUER|"
            r"URZAD SKARBOWY|PODATEK|IMPOT|IMPOTS|IMPUESTO|BELASTING|BELASTINGDIENST|F24|"
            r"AGENZIA DELLE ENTRATE|HMRC|REVENUE COMMISSIONERS|IRS)\b"
        ),
        "outflow",
        9,
        0.92,
        "Direct tax payment",
    ),
    _rule(
        "payroll",
        (
            r"\b(PAYROLL|SALARY|SALARIES|SALARIS|SALARI|WAGES|LOHN|LOHNE|GEHALT|GEHALTER|"
            r"SALAIRE|SALARIOS|SALARIO|WYNAGRODZENIE|WYNAGRODZENIA|PAYE|PAYSLIP|"
            r"BONUS PAYMENT|EMPLOYEE PAYMENT|STAFF COST|SOCIAL SECURITY|URSSAF|INPS|ZUS|KRUS)\b"
        ),
        "outflow",
        10,
        0.95,
        "Employee salary, payroll or social contribution payment",
    ),
    _rule(
        "interest_expense",
        (
            r"\b(INTEREST|INTERESES|RENTE|ZINSEN|INTERETS|INTERESSI|ODSETKI|"
            r"FINANCING CHARGE|BANK INTEREST|INTEREST PAYMENT|INTEREST CHARGED)\b"
        ),
        "outflow",
        11,
        0.92,
        "Interest or financing charge",
    ),
    _rule(
        "loan_repayment",
        (
            r"\b(LOAN REPAYMENT|LOAN INSTAL|LOAN INSTALL|INSTALMENT|INSTALLMENT|RATA|RATY|"
            r"KREDITRATE|TILGUNG|LENING|PRESTAMO CUOTA|CUOTA PRESTAMO|RATA PRESTITO|"
            r"PRET IMMOBILIER|MUTUO|MORTGAGE|LEASE PAYMENT|LEASING RATE|FINANCE PAYMENT|"
            r"DEBT SERVICE|CAPITAL REPAYMENT)\b"
        ),
        "outflow",
        12,
        0.92,
        "Loan principal or lease instalment",
    ),
    _rule(
        "bank_fees",
        (
            r"\b(FEE|FEES|GEBUEHR|GEBUHR|FRAIS|COMISION|COMMISSION|COMMISSIONE|SPESE|"
            r"OPLATA|OPLATY|KONTOFUEHRUNGSGEBUEHR|KONTOGEBUEHR|BUCHUNGSGEBUEHR|"
            r"BANK KOSTEN|SERVICE CHARGE|ACCOUNT MAINTENANCE|CARD FEE|MONTHLY FEE|"
            r"TRANSACTION FEE)\b"
        ),
        "outflow",
        13,
        0.9,
        "Bank or payment-provider fee",
    ),
    _rule(
        "gambling",
        (
            r"\b(BET|BETS|BET365|CASINO|LOTTERY|LOTERIJ|LOTTO|POKER|BOOKMAKER|"
            r"WILLIAM HILL|PADDY POWER|UNIBET|SPORTINGBET|GAMBLING)\b"
        ),
        "outflow",
        14,
        0.95,
        "Gambling or betting transaction",
    ),
    _rule(
        "insurance",
        (
            r"\b(INSURANCE|VERSICHERUNG|VERZEKERING|SEGURO|ASSURANCE|ASSICURAZIONE|"
            r"UBEZPIECZENIE|INSURANCE PREMIUM|POLICY PREMIUM)\b"
        ),
        "outflow",
        15,
        0.92,
        "Insurance premium",
    ),
    _rule(
        "rent",
        (
            r"\b(RENT|HUUR|ALQUILER|MIETE|MIETZAHLUNG|LOYER|AFFITTO|CZYNSZ|"
            r"PROPERTY RENT|OFFICE RENT|WAREHOUSE RENT|LANDLORD)\b"
        ),
        "outflow",
        16,
        0.92,
        "Rent or property lease",
    ),
    _rule(
        "utilities",
        (
            r"\b(ELECTRIC|ELECTRICITY|STROM|STROMRECHNUNG|STROMKOSTEN|ENERGIE|ENERGIEKOSTEN|"
            r"ENERGY BILL|WATER BILL|GAS BILL|UTILIT|INTERNET|BROADBAND|TELECOM|TELEKOM|"
            r"VODAFONE|ORANGE|T MOBILE|TELEFONICA|VATTENFALL|ENECO|PGE|ENEA|EDF|ENGIE|"
            r"E ON|IBERDROLA|ENEL|A2A|VEOLIA|SUEZ|MUNICIPAL|WASTE MANAGEMENT)\b"
        ),
        "outflow",
        17,
        0.9,
        "Utility, telecom or municipal bill",
    ),
    _rule(
        "loan_proceeds",
        (
            r"\b(LOAN DISBURSEMENT|LOAN PROCEEDS|DRAWDOWN|CREDIT FACILITY|KREDITAUSZAHLUNG|"
            r"PRET ACCORDE|MUTUO EROGATO|PRESTAMO CONCEDIDO|FINANCING RECEIVED|"
            r"SHAREHOLDER LOAN|FACILITY DRAWING)\b"
        ),
        "inflow",
        18,
        0.95,
        "Loan drawdown or financing received",
    ),
    _rule(
        "equity_injection",
        (r"\b(CAPITAL INJECTION)\b",),
        "inflow",
        18,
        0.95,
        "Owner or shareholder equity received",
    ),
    _rule(
        "interest_income",
        (
            r"\b(INTEREST CREDIT|INTEREST PAID BY BANK|ZINSGUTSCHRIFT|INTERETS CREDITEURS|"
            r"INTERESSI ATTIVI|CREDIT INTEREST|DEPOSIT INTEREST)\b"
        ),
        "inflow",
        19,
        0.92,
        "Interest credited by the bank",
    ),
    _rule(
        "refund",
        (
            r"\b(REFUND|REIMBURSEMENT|RUECKERSTATTUNG|RUCKERSTATTUNG|REMBOURSEMENT|RIMBORSO|"
            r"RESTITUZIONE|RESTITUTIE|DEVOLUCION|ZWROT|RETURNED FUNDS|CHARGEBACK RECEIVED|"
            r"TAX REFUND)\b"
        ),
        "inflow",
        20,
        0.9,
        "Refund or reimbursement received",
    ),
    _rule(
        "revenue",
        (
            r"\b(INVOICE|INVOICES|FAKTURA|FAKTUR|RECHNUNG|FACTURE|FATTURA|NOTA FISCAL|"
            r"SALES|SALE PROCEEDS|CUSTOMER PAYMENT|PAYMENT FROM|PAYMENT RECEIVED|RECEIPT|"
            r"STRIPE|PAYPAL|ADYEN|MOLLIE|SUMUP|SHOPIFY|SQUARE|KLARNA|ZAHLUNGSEINGANG|"
            r"GUTSCHRIFT|ONTVANGEN|TRANSFERENCIA RECIBIDA|WPLATA|PRZELEW PRZYCHODZACY|"
            r"VIREMENT RECU|ENCAISSEMENT|INCASSO|RIMESSA|BONIFICO RICEVUTO|ACCREDITO|"
            r"CUSTOMER|CLIENT|KUNDE|KLIENT|CLIENTE)\b"
        ),
        "inflow",
        25,
        0.85,
        "Customer receipt or sales settlement",
    ),
    _rule(
        "supplier",
        (
            r"\b(SUPPLIER|LIEFERANT|FOURNISSEUR|FACTUUR|FACTURA|INVOICE|RECHNUNG|FACTURE|"
            r"FATTURA|FAKTURA|FORNITORE|DOSTAWCA|PURCHASE|PURCHASE ORDER|WHOLESALE|"
            r"GROCERY|RAW MATERIAL|INVENTORY|STOCK|METRO|SELGROS|ALIBABA|MAKRO|"
            r"BOOKER|COSTCO|TRADE SUPPLIER)\b"
        ),
        "outflow",
        28,
        0.9,
        "Supplier, wholesale or inventory purchase",
    ),
    _rule(
        "card_payment",
        (
            r"\b(CARD PAYMENT|CARD PURCHASE|POS PAYMENT|POS PURCHASE|DEBIT CARD|CREDIT CARD|"
            r"CONTACTLESS|PIN BETALING|PAGO TARJETA|KARTENZAHLUNG|KARTENUMSATZ|"
            r"PAIEMENT CARTE|PAGAMENTO CARTA|PLATNOSC KARTA|MAESTRO|VISA|MASTERCARD|AMEX)\b"
        ),
        "outflow",
        30,
        0.85,
        "Card or point-of-sale payment",
    ),
    _rule(
        "cash_withdrawal",
        (
            r"\b(ATM|CASH WITHDRAWAL|CASH WITHDRAW|BARGELD|GELDAUTOMAT|GELDAUTOMAAT|"
            r"RETRAIT|RETRAIT DAB|RETIRADA|CAJERO|PRELIEVO|BANKOMAT|WYPLATA|"
            r"WYPLATA Z BANKOMATU)\b"
        ),
        "outflow",
        32,
        0.95,
        "Cash withdrawal",
    ),
    _rule(
        "transfer_out",
        (
            r"\b(TRANSFER TO|OUTGOING TRANSFER|SEPA TRANSFER|UBERWEISUNG|VIREMENT EMIS|"
            r"VIREMENT INTERNE|BONIFICO INTERNO|BONIFICO|OVERBOEKING|PRZELEW|TRANSFER)\b"
        ),
        "outflow",
        40,
        0.7,
        "Outgoing transfer",
    ),
    _rule(
        "transfer_in",
        (
            r"\b(TRANSFER FROM|INCOMING TRANSFER|SEPA CREDIT|UBERWEISUNG|VIREMENT RECU|"
            r"BONIFICO RICEVUTO|OVERBOEKING|PRZELEW|TRANSFER)\b"
        ),
        "inflow",
        40,
        0.7,
        "Incoming transfer",
    ),
)


@dataclass(frozen=True)
class CategoryMatch:
    """Result of categorising one transaction."""

    category: str
    confidence: float
    method: str
    rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "method": self.method,
            "rule": self.rule,
        }


def normalize_text(text: str) -> str:
    """Accent-fold, upper-case and strip reference numbers from a bank narrative.

    The output is intentionally lossy: it removes punctuation and long digit strings so that
    ``"FAKTURA 2024/00123"`` and ``"FAKTURA 2025/00999"`` match the same rule.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.upper())
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = re.sub(r"\b\d+[A-Z]+\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TextCategorizer:
    """Lightweight character n-gram classifier used as an optional fallback.

    Character n-grams are robust to the abbreviations, accents and concatenated tokens common in
    bank narratives. The model is deliberately small: it is a fallback, not the primary engine.
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_features: int = 30_000,
        C: float = 4.0,
        random_state: int = 42,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.C = C
        self.random_state = random_state
        self._vectorizer: Any = None
        self._model: Any = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _prepare(texts: Any, directions: Any = None) -> list[str]:
        texts = pd.Series(texts).fillna("").astype(str)
        if directions is None:
            return [normalize_text(t) for t in texts]
        dirs = pd.Series(directions).fillna("").astype(str).str.upper()
        return [f"{d} {normalize_text(t)}" for d, t in zip(dirs, texts)]

    def fit(self, texts: Any, labels: Any, directions: Any = None) -> TextCategorizer:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        labels = pd.Series(labels).astype(str)
        if labels.nunique() < 2:
            raise ValueError("TextCategorizer needs at least two distinct categories.")
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(self._prepare(texts, directions))
        self._model = LogisticRegression(
            max_iter=1_000,
            C=self.C,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._model.fit(X, labels)
        self.classes_ = self._model.classes_
        return self

    def predict(self, texts: Any, directions: Any = None) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(self._vectorizer.transform(self._prepare(texts, directions)))

    def predict_proba(self, texts: Any, directions: Any = None) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(
            self._vectorizer.transform(self._prepare(texts, directions))
        )

    def _check_fitted(self) -> None:
        if self._model is None or self._vectorizer is None:
            raise RuntimeError("TextCategorizer is not fitted. Call fit() first.")


class TransactionCategorizer:
    """Rule-first transaction categoriser with an optional learned fallback.

    Args:
        rules: Rule set to use. Defaults to :data:`DEFAULT_RULES`.
        fallback_inflow: Category for unmatched credits.
        fallback_outflow: Category for unmatched debits.
        ml_min_confidence: Minimum classifier probability required to override a fallback.
        random_state: Seed for the optional text classifier.

    Example:
        >>> cat = TransactionCategorizer()
        >>> cat.categorize("LOHN MAI 2026", "outflow").category
        'payroll'
        >>> cat.categorize("SEPA CREDIT TRANSFER FROM ACME INVOICE 42", "inflow").category
        'revenue'
    """

    def __init__(
        self,
        rules: tuple[CategoryRule, ...] | list[CategoryRule] | None = None,
        *,
        fallback_inflow: str = "other_inflow",
        fallback_outflow: str = "other_outflow",
        ml_min_confidence: float = 0.55,
        random_state: int = 42,
    ) -> None:
        if fallback_inflow not in INFLOW_CATEGORIES:
            raise ValueError(
                f"fallback_inflow must be an inflow category, got {fallback_inflow!r}."
            )
        if fallback_outflow not in OUTFLOW_CATEGORIES:
            raise ValueError(
                f"fallback_outflow must be an outflow category, got {fallback_outflow!r}."
            )
        self.rules = tuple(sorted(rules or DEFAULT_RULES, key=lambda r: (r.priority, r.category)))
        self.fallback_inflow = fallback_inflow
        self.fallback_outflow = fallback_outflow
        self.ml_min_confidence = ml_min_confidence
        self.random_state = random_state
        self.ml_model: TextCategorizer | None = None
        self._rule_hits: dict[str, int] = {}

    def categorize(
        self,
        description: str,
        direction: str = "any",
        counterparty: str = "",
        *,
        use_ml: bool = True,
    ) -> CategoryMatch:
        """Categorise one transaction narrative.

        Args:
            description: Bank narrative, memo or remittance information.
            direction: ``"inflow"`` or ``"outflow"``. ``"any"`` disables direction filtering.
            counterparty: Optional merchant or counterparty name, appended to the text.
            use_ml: Use the trained fallback model when no rule matches.

        Returns:
            A :class:`CategoryMatch` with category, confidence, method and matched rule.
        """
        direction = direction if direction in {"inflow", "outflow"} else "any"
        text = normalize_text(f"{description} {counterparty}")
        for rule in self.rules:
            matched = rule.matches(text, direction)
            if matched:
                self._rule_hits[rule.category] = self._rule_hits.get(rule.category, 0) + 1
                return CategoryMatch(rule.category, rule.confidence, "rule", matched)
        if use_ml and self.ml_model is not None and text:
            proba = self.ml_model.predict_proba([text], [direction])[0]
            idx = int(np.argmax(proba))
            confidence = float(proba[idx])
            category = str(self.ml_model.classes_[idx])
            if confidence >= self.ml_min_confidence and category in CATEGORY_DESCRIPTIONS:
                return CategoryMatch(category, confidence, "ml", None)
        fallback = self.fallback_inflow if direction != "outflow" else self.fallback_outflow
        return CategoryMatch(fallback, 0.3 if text else 0.0, "fallback", None)

    def categorize_frame(
        self,
        transactions: pd.DataFrame,
        *,
        description_col: str = "description",
        counterparty_col: str = "counterparty",
        direction_col: str = "direction",
        amount_col: str = "amount",
        category_col: str = "category",
        overwrite: bool = False,
        use_ml: bool = True,
    ) -> pd.DataFrame:
        """Add ``category``, ``category_confidence`` and ``category_method`` columns.

        Existing labels in ``category_col`` are preserved unless ``overwrite=True``. When the input
        is a raw table without a direction column, direction is inferred from the sign of
        ``amount_col``.
        """
        if description_col not in transactions.columns:
            raise KeyError(f"Missing description column {description_col!r}.")
        out = transactions.copy()
        if direction_col in out.columns:
            directions = out[direction_col].astype(str).str.lower()
            directions = directions.where(directions.isin(["inflow", "outflow"]), "any")
        elif amount_col in out.columns:
            amounts = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
            directions = pd.Series(np.where(amounts >= 0, "inflow", "outflow"), index=out.index)
        else:
            directions = pd.Series("any", index=out.index)
        counterparties = (
            out[counterparty_col].fillna("").astype(str)
            if counterparty_col in out.columns
            else pd.Series("", index=out.index)
        )
        matches = [
            self.categorize(desc, direction, counterparty, use_ml=use_ml)
            for desc, direction, counterparty in zip(
                out[description_col], directions, counterparties
            )
        ]
        predicted = pd.Series([m.category for m in matches], index=out.index, dtype="object")
        if category_col in out.columns and not overwrite:
            existing = out[category_col]
            keep = existing.notna() & (existing.astype(str).str.strip() != "")
            out["category"] = existing.where(keep, predicted)
        else:
            out["category"] = predicted
        out["category_confidence"] = [m.confidence for m in matches]
        out["category_method"] = [m.method for m in matches]
        return out

    def fit(
        self,
        transactions: pd.DataFrame,
        *,
        category_col: str = "category",
        description_col: str = "description",
        direction_col: str = "direction",
        min_samples_per_class: int = 3,
    ) -> TransactionCategorizer:
        """Train the optional ML fallback from a labelled transaction table.

        Rows without a label are ignored. Classes with fewer than ``min_samples_per_class`` rows
        are dropped to avoid a brittle classifier.
        """
        if category_col not in transactions.columns:
            raise KeyError(f"Missing label column {category_col!r}.")
        labelled = transactions[transactions[category_col].notna()].copy()
        labels = labelled[category_col].astype(str)
        counts = labels.value_counts()
        keep = counts[counts >= min_samples_per_class].index
        labelled = labelled[labels.isin(keep)]
        labels = labelled[category_col].astype(str)
        if labels.nunique() < 2 or len(labelled) < 10:
            raise ValueError(
                "Need at least 10 labelled rows and two categories with "
                f"{min_samples_per_class}+ examples each to train the fallback."
            )
        directions = labelled[direction_col] if direction_col in labelled.columns else None
        self.ml_model = TextCategorizer(random_state=self.random_state).fit(
            labelled[description_col], labels, directions
        )
        return self

    def fit_ml(self, transactions: pd.DataFrame, **kwargs: Any) -> TransactionCategorizer:
        """Alias for :meth:`fit`."""
        return self.fit(transactions, **kwargs)

    @property
    def rule_hits(self) -> dict[str, int]:
        """Number of times each category was assigned by a rule since construction."""
        return dict(self._rule_hits)

    def category_summary(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate count, total amount and share of absolute flow by category."""
        if "category" not in transactions.columns:
            transactions = self.categorize_frame(transactions)
        grouped = (
            transactions.assign(abs_amount=transactions["amount"].abs())
            .groupby("category", observed=True)
            .agg(
                count=("amount", "size"),
                total_amount=("amount", "sum"),
                abs_amount=("abs_amount", "sum"),
                mean_amount=("amount", "mean"),
            )
            .sort_values("abs_amount", ascending=False)
        )
        total = grouped["abs_amount"].sum()
        grouped["share_of_flow"] = grouped["abs_amount"] / total if total else 0.0
        return grouped.reset_index()
