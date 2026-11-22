"""Tests for multilingual transaction categorisation."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from opencashflow.features.categorizer import (
    CATEGORY_DESCRIPTIONS,
    INFLOW_CATEGORIES,
    RETURNED_PAYMENT_PATTERN,
    TransactionCategorizer,
    normalize_text,
)

CASES = [
    ("revenue", "inflow", "SEPA CREDIT TRANSFER FROM ACME GMBH INVOICE 2024-001"),
    ("revenue", "inflow", "SEPA GUTSCHRIFT RECHNUNG 202401 VON ACME"),
    ("revenue", "inflow", "VIREMENT RECU FACTURE 123 DE ACME"),
    ("revenue", "inflow", "PRZELEW PRZYCHODZACY FAKTURA 123 OD ACME"),
    ("revenue", "inflow", "BONIFICO RICEVUTO FATTURA 123 DA ACME"),
    ("revenue", "inflow", "ONTVANGEN FACTUUR 123 VAN ACME"),
    ("revenue", "inflow", "TRANSFERENCIA RECIBIDA FACTURA 123 DE ACME"),
    ("payroll", "outflow", "LOHN MAI 2024"),
    ("payroll", "outflow", "SALAIRE MAI"),
    ("payroll", "outflow", "WYNAGRODZENIE MAJ"),
    ("payroll", "outflow", "SALARIO MAGGIO"),
    ("payroll", "outflow", "SALARIS MEI"),
    ("rent", "outflow", "MIETE BUERO BERLIN"),
    ("rent", "outflow", "LOYER BUREAU PARIS"),
    ("rent", "outflow", "CZYNSZ BIURO"),
    ("rent", "outflow", "AFFITTO UFFICIO"),
    ("rent", "outflow", "HUUR KANTOOR"),
    ("rent", "outflow", "ALQUILER OFICINA"),
    ("utilities", "outflow", "STROMRECHNUNG VATTENFALL"),
    ("utilities", "outflow", "FACTURE EDF ENERGIE"),
    ("utilities", "outflow", "FAKTURA ENERGIA PGE"),
    ("utilities", "outflow", "FATTURA ENEL ENERGIA"),
    ("utilities", "outflow", "ENERGIE FACTUUR"),
    ("utilities", "outflow", "FACTURA IBERDROLA ENERGIA"),
    ("tax", "outflow", "FINANZAMT STEUER 2024"),
    ("tax", "outflow", "IMPOT SOCIETES 2024"),
    ("tax", "outflow", "PODATEK DOCHODOWY 2024"),
    ("tax", "outflow", "F24 TASSE 2024"),
    ("tax", "outflow", "BELASTINGDIENST 2024"),
    ("vat", "outflow", "UMSATZSTEUER JAN"),
    ("vat", "outflow", "TVA JANVIER"),
    ("vat", "outflow", "VAT JANUARY"),
    ("vat", "outflow", "IVA GENNAIO"),
    ("vat", "outflow", "BTW JANUARI"),
    ("supplier", "outflow", "RECHNUNG LIEFERANT ACME"),
    ("supplier", "outflow", "FACTURE FOURNISSEUR ACME"),
    ("supplier", "outflow", "FAKTURA DOSTAWCA ACME"),
    ("supplier", "outflow", "FATTURA FORNITORE ACME"),
    ("supplier", "outflow", "FACTUUR LEVERANCIER ACME"),
    ("supplier", "outflow", "FACTURA PROVEEDOR ACME"),
    ("loan_repayment", "outflow", "KREDITRATE DARLEHEN"),
    ("loan_repayment", "outflow", "PRET IMMOBILIER ECHEANCE"),
    ("loan_repayment", "outflow", "RATA KREDYTU"),
    ("loan_repayment", "outflow", "RATA MUTUO"),
    ("loan_repayment", "outflow", "LENING AFLOSSING"),
    ("loan_repayment", "outflow", "PRESTAMO CUOTA"),
    ("interest_expense", "outflow", "ZINSEN DARLEHEN"),
    ("interest_expense", "outflow", "INTERETS PRET"),
    ("interest_expense", "outflow", "ODSETKI KREDYT"),
    ("interest_expense", "outflow", "INTERESSI MUTUO"),
    ("interest_expense", "outflow", "RENTE LENING"),
    ("interest_expense", "outflow", "INTERESES PRESTAMO"),
    ("bank_fees", "outflow", "KONTOFUEHRUNGSGEBUEHR"),
    ("bank_fees", "outflow", "FRAIS BANCAIRES"),
    ("bank_fees", "outflow", "OPLATA BANKOWA"),
    ("bank_fees", "outflow", "SPESE BANCARIE"),
    ("bank_fees", "outflow", "BANK KOSTEN"),
    ("bank_fees", "outflow", "COMISION BANCARIA"),
    ("insurance", "outflow", "VERSICHERUNG PRAEMIE"),
    ("insurance", "outflow", "ASSURANCE PREMIUM"),
    ("insurance", "outflow", "UBEZPIECZENIE"),
    ("insurance", "outflow", "ASSICURAZIONE PREMIO"),
    ("insurance", "outflow", "VERZEKERING PREMIE"),
    ("insurance", "outflow", "SEGURO PRIMA"),
    ("card_payment", "outflow", "KARTENZAHLUNG MEDIAMARKT"),
    ("card_payment", "outflow", "PAIEMENT CARTE MEDIAMARKT"),
    ("card_payment", "outflow", "PLATNOSC KARTA MEDIAMARKT"),
    ("card_payment", "outflow", "PAGAMENTO CARTA MEDIAMARKT"),
    ("card_payment", "outflow", "PIN BETALING MEDIAMARKT"),
    ("card_payment", "outflow", "PAGO TARJETA MEDIAMARKT"),
    ("cash_withdrawal", "outflow", "GELDAUTOMAT ABHEBUNG"),
    ("cash_withdrawal", "outflow", "RETRAIT DAB"),
    ("cash_withdrawal", "outflow", "WYPLATA Z BANKOMATU"),
    ("cash_withdrawal", "outflow", "PRELIEVO BANCOMAT"),
    ("cash_withdrawal", "outflow", "GELDAUTOMAAT"),
    ("cash_withdrawal", "outflow", "RETIRADA CAJERO"),
    ("gambling", "outflow", "BET365 EINZAHLUNG"),
    ("gambling", "outflow", "CASINO EN LIGNE"),
    ("gambling", "outflow", "LOTTO"),
    ("gambling", "outflow", "LOTERIJ"),
    ("nsf", "outflow", "LASTSCHRIFT NICHT AUSGEFUEHRT"),
    ("nsf", "outflow", "PRELEVEMENT IMPAYE"),
    ("nsf", "outflow", "PRZELEW ODRZUCONY"),
    ("nsf", "outflow", "ADDEBITO RIFIUTATO"),
    ("nsf", "outflow", "TERUGGEBOEKT ONVOLDOENDE SALDO"),
    ("nsf", "outflow", "RECIBO DEVUELTO"),
    ("refund", "inflow", "RUECKERSTATTUNG LIEFERANT"),
    ("refund", "inflow", "REMBOURSEMENT FOURNISSEUR"),
    ("refund", "inflow", "ZWROT FAKTURA"),
    ("refund", "inflow", "RIMBORSO FORNITORE"),
    ("refund", "inflow", "RESTITUTIE LEVERANCIER"),
    ("refund", "inflow", "DEVOLUCION PROVEEDOR"),
]


@pytest.mark.parametrize(("expected", "direction", "description"), CASES)
def test_multilingual_categories(expected, direction, description):
    match = TransactionCategorizer().categorize(description, direction)
    assert match.category == expected, f"{description!r} -> {match.category} ({match.rule})"
    assert match.method == "rule"
    assert match.confidence > 0


def test_direction_aware_invoice():
    categorizer = TransactionCategorizer()
    assert categorizer.categorize("INVOICE ACME 123", "inflow").category == "revenue"
    assert categorizer.categorize("INVOICE ACME 123", "outflow").category == "supplier"


def test_internal_transfers():
    categorizer = TransactionCategorizer()
    assert (
        categorizer.categorize("INTERNAL TRANSFER FROM OWN ACCOUNT", "inflow").category
        == "transfer_in"
    )
    assert (
        categorizer.categorize("INTERNAL TRANSFER TO OWN ACCOUNT", "outflow").category
        == "transfer_out"
    )
    assert categorizer.categorize("VIREMENT INTERNE COMPTE", "inflow").category == "transfer_in"
    assert categorizer.categorize("BONIFICO INTERNO", "outflow").category == "transfer_out"


def test_unknown_direction_fallback():
    categorizer = TransactionCategorizer()
    inflow = categorizer.categorize("ZZZ RANDOM CREDIT", "inflow")
    outflow = categorizer.categorize("ZZZ RANDOM DEBIT", "outflow")
    assert inflow.category == "other_inflow"
    assert outflow.category == "other_outflow"
    assert inflow.method == "fallback"


def test_normalize_text_removes_reference_noise_and_accents():
    text = normalize_text("FAKTURA 2024/00123 für Müller & Söhne")
    assert "2024" not in text
    assert "00123" not in text
    assert "MULLER" in text
    assert "SOHNE" in text
    assert "AND" in text


def test_categorize_frame_preserves_labels_and_adds_metadata():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [-100.0, 200.0],
            "description": ["INVOICE ACME", "INVOICE ACME"],
            "category": ["payroll", None],
        }
    )
    categorizer = TransactionCategorizer()
    preserved = categorizer.categorize_frame(frame, overwrite=False)
    assert preserved.loc[0, "category"] == "payroll"
    assert preserved.loc[1, "category"] == "revenue"
    assert set(preserved["category_method"]) <= {"rule", "ml", "fallback"}
    overwritten = categorizer.categorize_frame(frame, overwrite=True)
    assert overwritten.loc[0, "category"] == "supplier"


def test_category_summary():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [-100.0, 200.0, -50.0],
            "description": ["RENT PAYMENT", "INVOICE ACME", "SALARY"],
        }
    )
    summary = TransactionCategorizer().category_summary(frame)
    assert {"category", "count", "total_amount", "share_of_flow"} <= set(summary.columns)
    assert summary["share_of_flow"].sum() == pytest.approx(1.0)


def test_ml_fallback_can_be_trained():
    rows = []
    for i in range(15):
        rows.append(
            {"description": f"ZZZ ALPHA VENDOR {i}", "category": "revenue", "direction": "inflow"}
        )
        rows.append(
            {"description": f"ZZZ BETA WAGES {i}", "category": "payroll", "direction": "outflow"}
        )
    labelled = pd.DataFrame(rows)
    categorizer = TransactionCategorizer(ml_min_confidence=0.0)
    categorizer.fit(labelled, min_samples_per_class=5)
    match = categorizer.categorize("ZZZ ALPHA PAYMENT", "inflow")
    assert match.method == "ml"
    assert match.category in {"revenue", "payroll"}


def test_every_category_has_description():
    assert set(CATEGORY_DESCRIPTIONS) >= {
        "revenue",
        "payroll",
        "rent",
        "nsf",
        "other_outflow",
    }
    assert all(CATEGORY_DESCRIPTIONS.values())


NSF_EXAMPLES = [
    ("DE", "LASTSCHRIFT NICHT AUSGEFUEHRT UNGENUEGENDE DECKUNG"),
    ("FR", "PRELEVEMENT IMPAYE"),
    ("PL", "PRZELEW ODRZUCONY"),
    ("IT", "ADDEBITO RIFIUTATO"),
    ("NL", "TERUGGEBOEKT ONVOLDOENDE SALDO"),
    ("ES", "RECIBO DEVUELTO"),
    ("EN", "RETURNED PAYMENT"),
]


@pytest.mark.parametrize(("language", "description"), NSF_EXAMPLES)
def test_returned_payment_examples_are_nsf_and_match_pattern(language, description):
    match = TransactionCategorizer().categorize(description, "outflow")
    assert match.category == "nsf", f"{language}: {description!r} -> {match.category}"
    assert re.search(RETURNED_PAYMENT_PATTERN, normalize_text(description)), language


def test_social_contributions_are_payroll():
    categorizer = TransactionCategorizer()
    assert categorizer.categorize("URSSAF BEITRAEGE MAI 2025", "outflow").category == "payroll"
    assert categorizer.categorize("ZUS SKLADKI", "outflow").category == "payroll"
    assert categorizer.categorize("FINANZAMT STEUER 2024", "outflow").category == "tax"


def test_capital_injection_is_equity_injection():
    categorizer = TransactionCategorizer()
    assert (
        categorizer.categorize("CAPITAL INJECTION SHAREHOLDER", "inflow").category
        == "equity_injection"
    )
    assert categorizer.categorize("SHAREHOLDER LOAN", "inflow").category == "loan_proceeds"
    assert "equity_injection" in CATEGORY_DESCRIPTIONS
    assert "equity_injection" in INFLOW_CATEGORIES


def test_categorize_without_direction_uses_direction_rules():
    categorizer = TransactionCategorizer()
    no_direction = categorizer.categorize("LOHN MAI 2026")
    with_outflow = categorizer.categorize("LOHN MAI 2026", "outflow")
    assert no_direction.category == "payroll"
    assert with_outflow.category == "payroll"


def test_categorize_without_direction_matches_inflow_category():
    categorizer = TransactionCategorizer()
    narrative = "SEPA CREDIT TRANSFER FROM ACME INVOICE 42"
    no_direction = categorizer.categorize(narrative)
    with_inflow = categorizer.categorize(narrative, "inflow")
    # The revenue rule (priority 25) fires before the transfer_in rule (priority 40).
    assert no_direction.category == "revenue"
    assert with_inflow.category == "revenue"


RETURN_VOCABULARY_CASES = [
    # (language, narrative, direction, expected category, marked as a return)
    ("EN", "RETURNED PAYMENT", "outflow", "nsf", True),
    ("EN", "REFUND CUSTOMER", "inflow", "refund", False),
    ("DE", "LASTSCHRIFT NICHT EINGELOEST", "outflow", "nsf", True),
    ("DE", "LASTSCHRIFT NICHT AUSGEFUEHRT", "outflow", "nsf", True),
    ("DE", "ZURUECKGEBUCHT LASTSCHRIFT", "outflow", "nsf", True),
    ("DE", "RUECKERSTATTUNG LIEFERANT", "inflow", "refund", False),
    ("FR", "PRELEVEMENT IMPAYE", "outflow", "nsf", True),
    ("FR", "PRELEVEMENT REJETE", "outflow", "nsf", True),
    ("FR", "REMBOURSEMENT FOURNISSEUR", "inflow", "refund", False),
    ("PL", "PRZELEW ODRZUCONY", "outflow", "nsf", True),
    ("PL", "ZWROT FAKTURA", "inflow", "refund", False),
    ("IT", "ADDEBITO RIFIUTATO", "outflow", "nsf", True),
    ("IT", "RIMBORSO FORNITORE", "inflow", "refund", False),
    ("NL", "TERUGGEBOEKT ONVOLDOENDE SALDO", "outflow", "nsf", True),
    ("NL", "GEWEIGERD BETALING", "outflow", "nsf", True),
    ("NL", "RESTITUTIE LEVERANCIER", "inflow", "refund", False),
    ("ES", "RECIBO DEVUELTO", "outflow", "nsf", True),
    ("ES", "PAGO RECHAZADO", "outflow", "nsf", True),
    ("ES", "DEVOLUCION PROVEEDOR", "inflow", "refund", False),
]


@pytest.mark.parametrize(
    ("language", "narrative", "direction", "expected", "is_return"), RETURN_VOCABULARY_CASES
)
def test_return_vocabulary_agrees_with_nsf_rule(
    language, narrative, direction, expected, is_return
):
    match = TransactionCategorizer().categorize(narrative, direction)
    assert match.category == expected, f"{language}: {narrative!r} -> {match.category}"
    is_nsf = match.category == "nsf"
    matches_pattern = bool(re.search(RETURNED_PAYMENT_PATTERN, normalize_text(narrative)))
    if is_nsf:
        assert matches_pattern, f"{language}: nsf without pattern match: {narrative!r}"
    if is_return:
        assert is_nsf and matches_pattern, f"{language}: return mismatch: {narrative!r}"


@pytest.mark.parametrize("narrative", ["ZWROT FAKTURA", "RESTITUTIE LEVERANCIER"])
def test_ambiguous_refund_words_are_not_nsf(narrative):
    # Direction "any" still checks the outflow-only nsf rule after the F5-A fix.
    match = TransactionCategorizer().categorize(narrative, "any")
    assert match.category == "refund", f"{narrative!r} -> {match.category}"
    assert match.method == "rule"


def test_zwrot_matches_pattern_but_is_a_refund():
    text = normalize_text("ZWROT FAKTURA")
    assert re.search(RETURNED_PAYMENT_PATTERN, text)
    assert TransactionCategorizer().categorize("ZWROT FAKTURA", "any").category == "refund"
