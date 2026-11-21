"""Synthetic SME transaction and portfolio generator.

Real transaction-level SME credit data is scarce because bank ledgers are confidential. This module
generates realistic multi-country ledgers with an explicit latent distress process, so that
cash-flow features have genuine predictive signal and the whole toolkit can be demonstrated and
tested without data-sharing agreements.

The generator is not a simulator of any specific bank. It is a controlled benchmark: revenue,
payroll, rent, supplier, tax, debt-service, fee, cash and distress events are produced with
country-specific narratives, then a default label is sampled from a risk score that depends on the
same latent stress that drives the ledger. This makes leakage-free experiments possible while
keeping the data-generating process documented.

Example:
    >>> from opencashflow.data.synthetic import generate_portfolio
    >>> portfolio = generate_portfolio(n_companies=20, months=18, seed=3)
    >>> portfolio.transactions.shape[0] > 100
    True
    >>> set(portfolio.labels.columns) >= {"entity_id", "default_flag", "pd_true"}
    True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from opencashflow.features.cashflow_features import FeatureConfig

SECTORS = (
    "Manufacturing",
    "Retail",
    "Wholesale",
    "Construction",
    "Transport",
    "Hospitality",
    "Professional services",
    "Healthcare",
    "Food processing",
    "Software",
)

COUNTRIES = ("DE", "FR", "PL", "IT", "NL", "ES")

CITIES: dict[str, tuple[str, ...]] = {
    "DE": ("Berlin", "Munich", "Hamburg", "Cologne", "Stuttgart"),
    "FR": ("Paris", "Lyon", "Marseille", "Toulouse", "Lille"),
    "PL": ("Warsaw", "Krakow", "Wroclaw", "Poznan", "Gdansk"),
    "IT": ("Milan", "Rome", "Turin", "Bologna", "Naples"),
    "NL": ("Amsterdam", "Rotterdam", "Utrecht", "Eindhoven", "Groningen"),
    "ES": ("Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"),
}

CUSTOMER_NAMES = (
    "Acme Trading",
    "Nordwind Logistics",
    "Blue Harbour Retail",
    "Vistula Components",
    "Alpine Foods",
    "Lumen Services",
    "Delta Systems",
    "Casa Verde",
    "Baltic Supplies",
    "Orion Manufacturing",
    "Meridian Health",
    "Atlas Construction",
    "Pinewood Studios",
    "Silverline Media",
    "Kestrel Engineering",
    "Novum Pharma",
    "Rheinland Handel",
    "Provence Distribution",
    "Lombardia Impianti",
    "Iberia Suministros",
)

SUPPLIER_NAMES = (
    "Euro Materials",
    "Hanseatic Wholesale",
    "Rhone Packaging",
    "Wisla Metals",
    "Adriatic Components",
    "Lowlands Paper",
    "Castilla Textiles",
    "Rhein Chemie",
    "Seine Logistics",
    "Baltic Plastics",
)

MERCHANTS = (
    "Office Depot",
    "Shell Fuel",
    "Deutsche Bahn",
    "Air France",
    "Booking.com",
    "Amazon Business",
    "IKEA",
    "MediaMarkt",
    "Leroy Merlin",
    "Local Restaurant",
    "Hotel Ibis",
    "Uber",
)

#: Localised narratives. Each template intentionally contains at least one token the default
#: categoriser recognises, which keeps the synthetic benchmark honest about the rule engine.
TEMPLATES: dict[str, dict[str, str]] = {
    "DE": {
        "revenue": "SEPA GUTSCHRIFT RECHNUNG {invoice} VON {customer}",
        "payroll": "LOHN {month}",
        "rent": "MIETE BUERO {city}",
        "utilities": "STROMRECHNUNG {provider}",
        "tax": "FINANZAMT STEUER {year}",
        "vat": "UMSATZSTEUER {month}",
        "supplier": "RECHNUNG LIEFERANT {supplier}",
        "loan": "KREDITRATE DARLEHEN",
        "interest": "ZINSEN DARLEHEN",
        "fees": "KONTOFUEHRUNGSGEBUEHR",
        "insurance": "VERSICHERUNG PRAEMIE",
        "card": "KARTENZAHLUNG {merchant}",
        "cash": "GELDAUTOMAT ABHEBUNG",
        "gambling": "BET365 EINZAHLUNG",
        "nsf": "LASTSCHRIFT NICHT AUSGEFUEHRT UNGENUEGENDE DECKUNG",
        "refund": "RUECKERSTATTUNG LIEFERANT {supplier}",
    },
    "FR": {
        "revenue": "VIREMENT RECU FACTURE {invoice} DE {customer}",
        "payroll": "SALAIRE {month}",
        "rent": "LOYER BUREAU {city}",
        "utilities": "FACTURE EDF ENERGIE",
        "tax": "IMPOT SOCIETES {year}",
        "vat": "TVA {month}",
        "supplier": "FACTURE FOURNISSEUR {supplier}",
        "loan": "PRET IMMOBILIER ECHEANCE",
        "interest": "INTERETS PRET",
        "fees": "FRAIS BANCAIRES",
        "insurance": "ASSURANCE PREMIUM",
        "card": "PAIEMENT CARTE {merchant}",
        "cash": "RETRAIT DAB",
        "gambling": "CASINO EN LIGNE",
        "nsf": "PRELEVEMENT IMPAYE",
        "refund": "REMBOURSEMENT FOURNISSEUR {supplier}",
    },
    "PL": {
        "revenue": "PRZELEW PRZYCHODZACY FAKTURA {invoice} OD {customer}",
        "payroll": "WYNAGRODZENIE {month}",
        "rent": "CZYNSZ BIURO {city}",
        "utilities": "FAKTURA ENERGIA PGE",
        "tax": "PODATEK DOCHODOWY {year}",
        "vat": "VAT {month}",
        "supplier": "FAKTURA DOSTAWCA {supplier}",
        "loan": "RATA KREDYTU",
        "interest": "ODSETKI KREDYT",
        "fees": "OPLATA BANKOWA",
        "insurance": "UBEZPIECZENIE",
        "card": "PLATNOSC KARTA {merchant}",
        "cash": "WYPLATA Z BANKOMATU",
        "gambling": "LOTTO",
        "nsf": "PRZELEW ODRZUCONY",
        "refund": "ZWROT FAKTURA {supplier}",
    },
    "IT": {
        "revenue": "BONIFICO RICEVUTO FATTURA {invoice} DA {customer}",
        "payroll": "SALARIO {month}",
        "rent": "AFFITTO UFFICIO {city}",
        "utilities": "FATTURA ENEL ENERGIA",
        "tax": "F24 TASSE {year}",
        "vat": "IVA {month}",
        "supplier": "FATTURA FORNITORE {supplier}",
        "loan": "RATA MUTUO",
        "interest": "INTERESSI MUTUO",
        "fees": "SPESE BANCARIE",
        "insurance": "ASSICURAZIONE PREMIO",
        "card": "PAGAMENTO CARTA {merchant}",
        "cash": "PRELIEVO BANCOMAT",
        "gambling": "CASINO",
        "nsf": "ADDEBITO RIFIUTATO",
        "refund": "RIMBORSO FORNITORE {supplier}",
    },
    "NL": {
        "revenue": "ONTVANGEN FACTUUR {invoice} VAN {customer}",
        "payroll": "SALARIS {month}",
        "rent": "HUUR KANTOOR {city}",
        "utilities": "ENERGIE FACTUUR {provider}",
        "tax": "BELASTINGDIENST {year}",
        "vat": "BTW {month}",
        "supplier": "FACTUUR LEVERANCIER {supplier}",
        "loan": "LENING AFLOSSING",
        "interest": "RENTE LENING",
        "fees": "BANK KOSTEN",
        "insurance": "VERZEKERING PREMIE",
        "card": "PIN BETALING {merchant}",
        "cash": "GELDAUTOMAAT",
        "gambling": "LOTERIJ",
        "nsf": "TERUGGEBOEKT ONVOLDOENDE SALDO",
        "refund": "RESTITUTIE LEVERANCIER {supplier}",
    },
    "ES": {
        "revenue": "TRANSFERENCIA RECIBIDA FACTURA {invoice} DE {customer}",
        "payroll": "SALARIO {month}",
        "rent": "ALQUILER OFICINA {city}",
        "utilities": "FACTURA IBERDROLA ENERGIA",
        "tax": "IMPUESTO SOCIEDADES {year}",
        "vat": "IVA {month}",
        "supplier": "FACTURA PROVEEDOR {supplier}",
        "loan": "PRESTAMO CUOTA",
        "interest": "INTERESES PRESTAMO",
        "fees": "COMISION BANCARIA",
        "insurance": "SEGURO PRIMA",
        "card": "PAGO TARJETA {merchant}",
        "cash": "RETIRADA CAJERO",
        "gambling": "CASINO",
        "nsf": "RECIBO DEVUELTO",
        "refund": "DEVOLUCION PROVEEDOR {supplier}",
    },
}


@dataclass(frozen=True)
class CompanyProfile:
    """Parameters controlling one synthetic SME ledger.

    The defaults describe a healthy EUR 150k/month company with 24 months of history. Change
    ``distress`` to move a company along the risk spectrum; it affects revenue growth, balance,
    supplier costs, NSF frequency and gambling activity.
    """

    entity_id: str = "SME0001"
    sector: str = "Manufacturing"
    country: str = "DE"
    currency: str = "EUR"
    monthly_revenue: float = 150_000.0
    revenue_growth: float = 0.003
    revenue_volatility: float = 0.12
    seasonality_amplitude: float = 0.12
    seasonality_peak_month: int = 11
    gross_margin: float = 0.35
    payroll_ratio: float = 0.24
    rent_ratio: float = 0.06
    utilities_ratio: float = 0.02
    supplier_ratio: float = 0.38
    tax_rate: float = 0.20
    vat_rate: float = 0.19
    debt_payment: float = 4_000.0
    debt_interest: float = 900.0
    starting_balance: float = 60_000.0
    n_customers: int = 12
    customer_concentration: float = 0.30
    payment_delay_days: int = 12
    distress: float = 0.10
    nsf_probability: float = 0.01
    gambling_ratio: float = 0.0
    cash_withdrawal_probability: float = 0.30
    start_date: str = "2023-01-01"

    def __post_init__(self) -> None:
        if not 0.0 <= self.distress <= 1.0:
            raise ValueError("distress must be between 0 and 1.")
        if not 0.0 <= self.customer_concentration <= 1.0:
            raise ValueError("customer_concentration must be between 0 and 1.")
        if self.monthly_revenue <= 0:
            raise ValueError("monthly_revenue must be positive.")
        if self.country not in TEMPLATES:
            raise ValueError(
                f"Unsupported country {self.country!r}; choose from {sorted(TEMPLATES)}."
            )


@dataclass
class SyntheticPortfolio:
    """A generated portfolio: transactions, borrower labels and the latent profiles."""

    transactions: pd.DataFrame
    labels: pd.DataFrame
    profiles: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_features(
        self,
        config: FeatureConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Compute cash-flow features and align the default labels.

        Returns:
            ``(X, y)`` where ``X`` is indexed by ``entity_id`` and ``y`` is the binary
            ``default_flag`` aligned to ``X``.
        """
        from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig

        cfg = config or FeatureConfig(window_days=730)
        extractor = CashFlowFeatureExtractor(config=cfg)
        X = extractor.fit_transform(self.transactions)
        y = (
            self.labels.set_index("entity_id")["default_flag"]
            .reindex(X.index)
            .astype(int)
            .rename("default_flag")
        )
        return X, y

    def summary(self) -> str:
        """One-line description of size and default rate."""
        rate = float(self.labels["default_flag"].mean()) if len(self.labels) else 0.0
        return (
            f"{len(self.labels)} companies, {len(self.transactions):,} transactions, "
            f"default rate {rate:.1%}, "
            f"{self.transactions['date'].min().date()} to {self.transactions['date'].max().date()}"
        )


def _transaction(
    date: pd.Timestamp,
    amount: float,
    description: str,
    counterparty: str,
    currency: str,
) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date).normalize(),
        "amount": round(float(amount), 2),
        "description": description,
        "counterparty": counterparty,
        "currency": currency,
    }


def _seasonal_factor(month: int, amplitude: float, peak_month: int) -> float:
    """Cosine seasonal curve with a peak in ``peak_month`` (1-12)."""
    return 1.0 + amplitude * float(np.cos(2.0 * np.pi * (month - peak_month) / 12.0))


def _customer_weights(rng: np.random.Generator, n: int, concentration: float) -> np.ndarray:
    if n <= 1:
        return np.ones(1)
    pareto = rng.pareto(1.4, n) + 1.0
    pareto = pareto / pareto.sum()
    equal = np.full(n, 1.0 / n)
    weights = (1.0 - concentration) * equal + concentration * pareto
    return weights / weights.sum()


def generate_company_transactions(
    profile: CompanyProfile | None = None,
    months: int = 24,
    seed: int = 42,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate one company's transaction ledger.

    Args:
        profile: Company parameters. Defaults to :class:`CompanyProfile`.
        months: Number of calendar months to generate.
        seed: Random seed. The same seed and profile always produce the same ledger.
        start_date: First month. Defaults to ``profile.start_date``.

    Returns:
        Canonical-style transaction rows with ``entity_id``, ``date``, signed ``amount``,
        ``currency``, ``description``, ``counterparty``, ``balance`` and ``transaction_id``.
        Categories are intentionally omitted so the categoriser is exercised end to end.
    """
    profile = profile or CompanyProfile()
    if months < 1:
        raise ValueError("months must be at least 1.")
    rng = np.random.default_rng(seed)
    template = TEMPLATES[profile.country]
    city = str(rng.choice(CITIES[profile.country]))
    provider = str(rng.choice(("Vattenfall", "E.ON", "Engie", "Iberdrola", "Enel", "Eneco")))
    start = pd.Timestamp(start_date or profile.start_date)
    start = (start.to_period("M").start_time).normalize()
    rows: list[dict[str, Any]] = []

    for t in range(months):
        month_start = (start + pd.DateOffset(months=t)).normalize()
        month_label = month_start.strftime("%b %Y")
        year = str(month_start.year)
        stress = profile.distress

        base = (
            profile.monthly_revenue
            * (1.0 + profile.revenue_growth) ** t
            * _seasonal_factor(
                month_start.month, profile.seasonality_amplitude, profile.seasonality_peak_month
            )
            * float(np.clip(rng.lognormal(0.0, profile.revenue_volatility), 0.55, 1.6))
            * (1.0 - 0.12 * max(0.0, stress - 0.5))
        )
        base = max(base, 1_000.0)

        # Customer receipts, split across a concentrated or diversified customer base.
        n_customers = max(1, int(profile.n_customers))
        customers = rng.choice(
            CUSTOMER_NAMES, size=n_customers, replace=n_customers > len(CUSTOMER_NAMES)
        )
        weights = _customer_weights(rng, n_customers, profile.customer_concentration)
        for i, (customer, weight) in enumerate(zip(customers, weights)):
            day = int(np.clip(3 + profile.payment_delay_days + rng.integers(-4, 5), 1, 27))
            amount = base * float(weight) * float(rng.lognormal(0.0, 0.08))
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=day - 1),
                    amount,
                    template["revenue"].format(
                        invoice=f"{month_start.year}{month_start.month:02d}{i + 1:03d}",
                        customer=customer,
                        month=month_label,
                        year=year,
                    ),
                    str(customer),
                    profile.currency,
                )
            )

        # Payroll, rent, utilities and recurring overheads.
        payroll = base * profile.payroll_ratio * float(rng.normal(1.0, 0.03))
        if stress > 0.75 and rng.random() < 0.15:
            payroll *= 0.75
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=int(rng.integers(23, 28))),
                -payroll,
                template["payroll"].format(month=month_label, year=year),
                "Employees",
                profile.currency,
            )
        )
        rent = profile.monthly_revenue * profile.rent_ratio * float(rng.normal(1.0, 0.01))
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=1),
                -rent,
                template["rent"].format(city=city),
                f"Landlord {city}",
                profile.currency,
            )
        )
        utilities = profile.monthly_revenue * profile.utilities_ratio * float(rng.normal(1.0, 0.12))
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=int(rng.integers(8, 14))),
                -utilities,
                template["utilities"].format(provider=provider),
                provider,
                profile.currency,
            )
        )

        # Supplier payments: a few invoices per month with Dirichlet weights.
        n_suppliers = int(rng.integers(2, 5))
        supplier_total = (
            base * profile.supplier_ratio * (1.0 + 0.25 * stress) * float(rng.normal(1.0, 0.08))
        )
        supplier_weights = rng.dirichlet(np.ones(n_suppliers) * 1.5)
        suppliers = rng.choice(SUPPLIER_NAMES, size=n_suppliers, replace=False)
        for supplier, weight in zip(suppliers, supplier_weights):
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(2, 27))),
                    -supplier_total * float(weight),
                    template["supplier"].format(supplier=supplier),
                    str(supplier),
                    profile.currency,
                )
            )

        # Quarterly profit tax, monthly VAT, quarterly insurance.
        if month_start.month % 3 == 0:
            profit = (
                base * profile.gross_margin
                - payroll
                - rent
                - utilities
                - profile.debt_interest
                - base * 0.002
            )
            tax = max(0.0, profit) * profile.tax_rate
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=20),
                    -tax,
                    template["tax"].format(year=year),
                    "Tax authority",
                    profile.currency,
                )
            )
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=15),
                    -profile.monthly_revenue * 0.01,
                    template["insurance"],
                    "Insurer",
                    profile.currency,
                )
            )
        vat = base * profile.vat_rate / (1.0 + profile.vat_rate) * 0.75
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=int(rng.integers(18, 23))),
                -vat,
                template["vat"].format(month=month_label, year=year),
                "Tax authority",
                profile.currency,
            )
        )

        # Debt service and bank fees.
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=5),
                -profile.debt_payment * float(rng.normal(1.0, 0.01)),
                template["loan"],
                "Lender",
                profile.currency,
            )
        )
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=5),
                -profile.debt_interest * float(rng.normal(1.0, 0.05)),
                template["interest"],
                "Lender",
                profile.currency,
            )
        )
        rows.append(
            _transaction(
                month_start + pd.Timedelta(days=int(rng.integers(26, 29))),
                -(25.0 + profile.monthly_revenue * 0.0008),
                template["fees"],
                "Bank",
                profile.currency,
            )
        )

        # Discretionary card spend, cash, gambling, refunds and distress events.
        for _ in range(int(rng.integers(2, 5))):
            merchant = str(rng.choice(MERCHANTS))
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    -profile.monthly_revenue * float(rng.uniform(0.002, 0.012)),
                    template["card"].format(merchant=merchant),
                    merchant,
                    profile.currency,
                )
            )
        if rng.random() < profile.cash_withdrawal_probability:
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    -float(rng.uniform(200.0, 1_500.0)),
                    template["cash"],
                    "ATM",
                    profile.currency,
                )
            )
        if profile.gambling_ratio > 0 and rng.random() < min(0.6, profile.gambling_ratio * 6.0):
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    -profile.monthly_revenue * float(rng.uniform(0.005, 0.025)),
                    template["gambling"],
                    "Bookmaker",
                    profile.currency,
                )
            )
        if rng.random() < 0.10:
            supplier = str(rng.choice(SUPPLIER_NAMES))
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    profile.monthly_revenue * float(rng.uniform(0.003, 0.015)),
                    template["refund"].format(supplier=supplier),
                    supplier,
                    profile.currency,
                )
            )
        if rng.random() < profile.nsf_probability:
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    -(profile.monthly_revenue * float(rng.uniform(0.005, 0.02)) + 25.0),
                    template["nsf"],
                    "Bank",
                    profile.currency,
                )
            )

        # Internal transfers net to zero but test the categoriser and internal-transfer features.
        if rng.random() < 0.12:
            amount = profile.monthly_revenue * float(rng.uniform(0.05, 0.18))
            day = month_start + pd.Timedelta(days=int(rng.integers(1, 28)))
            rows.append(
                _transaction(
                    day,
                    amount,
                    "INTERNAL TRANSFER FROM OWN ACCOUNT",
                    "Own account",
                    profile.currency,
                )
            )
            rows.append(
                _transaction(
                    day,
                    -amount,
                    "INTERNAL TRANSFER TO OWN ACCOUNT",
                    "Own account",
                    profile.currency,
                )
            )

        # A distressed company may draw new debt or receive owner support.
        if stress > 0.6 and t == max(1, months // 2):
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=12),
                    profile.monthly_revenue * 2.0,
                    "LOAN DISBURSEMENT",
                    "Lender",
                    profile.currency,
                )
            )
        if stress > 0.85 and rng.random() < 0.1:
            rows.append(
                _transaction(
                    month_start + pd.Timedelta(days=int(rng.integers(1, 28))),
                    profile.monthly_revenue * 0.5,
                    "CAPITAL INJECTION SHAREHOLDER",
                    "Shareholder",
                    profile.currency,
                )
            )

    df = pd.DataFrame(rows)
    df = df.loc[df["amount"].abs() >= 0.01]
    df = df.sort_values(["date", "description"], kind="stable").reset_index(drop=True)
    df["entity_id"] = profile.entity_id
    df["transaction_id"] = [f"{profile.entity_id}-{i:06d}" for i in range(len(df))]
    df["balance"] = profile.starting_balance + df["amount"].cumsum()
    df["amount"] = df["amount"].round(2)
    df["balance"] = df["balance"].round(2)
    return df[
        [
            "entity_id",
            "transaction_id",
            "date",
            "amount",
            "currency",
            "description",
            "counterparty",
            "balance",
        ]
    ]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _calibrate_intercept(scores: np.ndarray, target_rate: float) -> float:
    """Find an intercept so that mean sigmoid(score + b) equals the target default rate."""
    target_rate = float(np.clip(target_rate, 1e-4, 1.0 - 1e-4))
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        rate = float(np.mean(_sigmoid(scores + mid)))
        if rate < target_rate:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


_NSF_PATTERN = r"NSF|INSUFFICIENT|RETURNED|IMPAG|ODRZUC|RIFIUT|TERUG|DEVUELTO|REJECT"


def _ledger_outcomes(transactions: pd.DataFrame, profiles: list[CompanyProfile]) -> pd.DataFrame:
    """Summarise post-ledger distress outcomes used to generate default labels.

    The label process is deliberately grounded in observable outcomes: low or negative balances,
    consecutive negative cash-flow months, failed payments and falling inflows. This keeps the
    synthetic benchmark learnable from the transaction features without leaking the latent stress
    variable directly into the model.
    """
    tx = transactions.sort_values(["entity_id", "date"], kind="stable").copy()
    tx["month"] = tx["date"].dt.to_period("M")
    grouped = tx.groupby("entity_id", observed=True)
    min_balance = grouped["balance"].min()
    ending_balance = grouped["balance"].last()
    monthly_net = tx.groupby(["entity_id", "month"], observed=True)["amount"].sum()
    negative_ratio = monthly_net.lt(0).groupby(level=0).mean()
    inflows = (
        tx.loc[tx["amount"] > 0].groupby(["entity_id", "month"], observed=True)["amount"].sum()
    )

    def _inflow_growth(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        window = min(3, len(values) // 2)
        if window < 1:
            return 0.0
        prior = float(values[:window].mean())
        recent = float(values[-window:].mean())
        return (recent / prior - 1.0) if prior > 0 else 0.0

    inflow_growth = inflows.groupby(level=0).apply(_inflow_growth)
    nsf_count = (
        tx.loc[tx["description"].str.contains(_NSF_PATTERN, case=False, regex=True)]
        .groupby("entity_id", observed=True)
        .size()
    )
    frame = pd.DataFrame(
        {
            "min_balance": min_balance,
            "ending_balance": ending_balance,
            "negative_month_ratio": negative_ratio,
            "inflow_growth": inflow_growth,
            "nsf_count": nsf_count,
        }
    )
    frame = frame.reindex([p.entity_id for p in profiles])
    return frame.fillna({"negative_month_ratio": 0.0, "inflow_growth": 0.0, "nsf_count": 0.0})


def generate_portfolio(
    n_companies: int = 120,
    months: int = 24,
    default_rate: float = 0.12,
    seed: int = 42,
    start_date: str = "2023-01-01",
) -> SyntheticPortfolio:
    """Generate a portfolio of SME ledgers with learnable default labels.

    Args:
        n_companies: Number of borrowers.
        months: Months of history per borrower.
        default_rate: Target mean probability of default used to calibrate the label process.
            The realised rate is binomial around this value.
        seed: Master random seed.
        start_date: First month of the ledger.

    Returns:
        A :class:`SyntheticPortfolio` with transactions, labels and profiles.
    """
    if n_companies < 2:
        raise ValueError("n_companies must be at least 2.")
    rng = np.random.default_rng(seed)
    profiles: list[CompanyProfile] = []
    for i in range(n_companies):
        stress = float(rng.beta(2.0, 5.0))
        country = str(rng.choice(COUNTRIES))
        sector = str(rng.choice(SECTORS))
        monthly_revenue = float(
            np.clip(rng.lognormal(np.log(150_000.0), 0.55), 25_000.0, 2_000_000.0)
        )
        profile = CompanyProfile(
            entity_id=f"SME{i + 1:04d}",
            sector=sector,
            country=country,
            monthly_revenue=monthly_revenue,
            revenue_growth=float(rng.normal(0.004, 0.008) - 0.025 * stress),
            revenue_volatility=float(rng.uniform(0.06, 0.20) + 0.15 * stress),
            seasonality_amplitude=float(rng.uniform(0.05, 0.20)),
            seasonality_peak_month=int(rng.integers(1, 13)),
            gross_margin=float(np.clip(rng.uniform(0.22, 0.50) - 0.12 * stress, 0.08, 0.60)),
            payroll_ratio=float(rng.uniform(0.18, 0.30)),
            rent_ratio=float(rng.uniform(0.04, 0.09)),
            utilities_ratio=float(rng.uniform(0.01, 0.03)),
            supplier_ratio=float(rng.uniform(0.30, 0.45)),
            debt_payment=float(monthly_revenue * rng.uniform(0.01, 0.06) * (1.0 + 0.8 * stress)),
            debt_interest=float(monthly_revenue * rng.uniform(0.001, 0.008) * (1.0 + stress)),
            starting_balance=float(
                monthly_revenue * rng.uniform(0.15, 1.20) * (1.0 - 0.5 * stress)
            ),
            n_customers=int(rng.integers(4, 25)),
            customer_concentration=float(rng.uniform(0.15, 0.60)),
            payment_delay_days=int(rng.integers(5, 25)),
            distress=stress,
            nsf_probability=float(0.003 + 0.06 * stress),
            gambling_ratio=float(0.03 * stress),
            cash_withdrawal_probability=float(rng.uniform(0.15, 0.45)),
            start_date=start_date,
        )
        profiles.append(profile)

    frames = [
        generate_company_transactions(profile, months=months, seed=seed * 10_000 + i)
        for i, profile in enumerate(profiles)
    ]
    transactions = pd.concat(frames, ignore_index=True)

    outcomes = _ledger_outcomes(transactions, profiles)
    stress = np.array([p.distress for p in profiles], dtype=float)
    monthly_revenue = np.array([p.monthly_revenue for p in profiles], dtype=float)
    min_balance_ratio = outcomes["min_balance"].to_numpy(dtype=float) / monthly_revenue
    ending_balance_ratio = outcomes["ending_balance"].to_numpy(dtype=float) / monthly_revenue
    negative_ratio = outcomes["negative_month_ratio"].clip(0.0, 1.0).to_numpy(dtype=float)
    inflow_growth = outcomes["inflow_growth"].clip(-1.0, 1.0).to_numpy(dtype=float)
    nsf_count = outcomes["nsf_count"].to_numpy(dtype=float)
    dsr = np.array([p.debt_payment for p in profiles], dtype=float) / monthly_revenue
    scores = (
        2.2 * stress
        + 2.5 * np.maximum(0.0, -min_balance_ratio)
        + 1.5 * np.maximum(0.0, -ending_balance_ratio)
        + 1.8 * negative_ratio
        + 1.2 * np.minimum(nsf_count, 5.0) / 5.0
        + 1.2 * np.maximum(0.0, dsr - 0.04) * 10.0
        + 1.5 * np.maximum(0.0, -inflow_growth)
    )
    intercept = _calibrate_intercept(scores, default_rate)
    pd_true = np.asarray(_sigmoid(scores + intercept), dtype=float)
    labels = rng.binomial(1, pd_true)
    if labels.min() == labels.max():
        # Keep tiny demo portfolios usable: default the riskiest company and spare the safest.
        order = np.argsort(pd_true)
        labels = np.zeros_like(labels)
        labels[order[-1]] = 1

    label_frame = pd.DataFrame(
        {
            "entity_id": [p.entity_id for p in profiles],
            "default_flag": labels.astype(int),
            "pd_true": pd_true,
            "stress": [p.distress for p in profiles],
            "sector": [p.sector for p in profiles],
            "country": [p.country for p in profiles],
            "monthly_revenue": [p.monthly_revenue for p in profiles],
            "revenue_growth": [p.revenue_growth for p in profiles],
            "starting_balance": [p.starting_balance for p in profiles],
            "nsf_probability": [p.nsf_probability for p in profiles],
            "debt_payment": [p.debt_payment for p in profiles],
            "gross_margin": [p.gross_margin for p in profiles],
            "min_balance_ratio": min_balance_ratio,
            "ending_balance_ratio": ending_balance_ratio,
            "negative_month_ratio": negative_ratio,
            "inflow_growth": inflow_growth,
            "nsf_count": nsf_count,
        }
    )
    profile_frame = pd.DataFrame([asdict(p) for p in profiles])
    metadata = {
        "n_companies": n_companies,
        "months": months,
        "default_rate_target": default_rate,
        "default_rate_realised": float(label_frame["default_flag"].mean()),
        "seed": seed,
        "start_date": start_date,
        "generator": "opencashflow.data.synthetic.generate_portfolio",
    }
    return SyntheticPortfolio(transactions, label_frame, profile_frame, metadata)


def generate_credit_dataset(
    n_companies: int = 120,
    months: int = 24,
    default_rate: float = 0.12,
    seed: int = 42,
    window_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, SyntheticPortfolio]:
    """Generate a portfolio and return ``(X, y, portfolio)`` ready for modelling."""
    portfolio = generate_portfolio(
        n_companies=n_companies,
        months=months,
        default_rate=default_rate,
        seed=seed,
    )
    from opencashflow.features.cashflow_features import FeatureConfig

    config = FeatureConfig(window_days=window_days or months * 31)
    X, y = portfolio.to_features(config)
    return X, y, portfolio


__all__ = [
    "TEMPLATES",
    "CompanyProfile",
    "SyntheticPortfolio",
    "generate_company_transactions",
    "generate_credit_dataset",
    "generate_portfolio",
]
