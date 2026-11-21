"""Tests for multilingual transaction categorisation."""

from __future__ import annotations

import pandas as pd
import pytest

from opencashflow.features.categorizer import (
    CATEGORY_DESCRIPTIONS,
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
