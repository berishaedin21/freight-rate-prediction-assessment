from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT / "validation_metrics.json").read_text(encoding="utf-8"))
chart = ROOT / "results" / "scorer_results" / "candidate_december.png"
out = ROOT / "assessment_report.pdf"

navy = colors.HexColor("#123047")
teal = colors.HexColor("#064A56")
light = colors.HexColor("#EAF1F3")
muted = colors.HexColor("#52666E")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=navy, alignment=TA_LEFT, spaceAfter=6))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=10, leading=14, textColor=muted, spaceAfter=16))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=teal, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9.4, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=8, leading=10, textColor=muted))
styles.add(ParagraphStyle(name="Kpi", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=navy, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="KpiLabel", parent=styles["BodyText"], fontSize=8, leading=10, textColor=muted, alignment=TA_CENTER))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6E0E3"))
    canvas.line(0.7 * inch, 0.52 * inch, 7.8 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(muted)
    canvas.drawString(0.7 * inch, 0.34 * inch, "Freight Rate Prediction - example assessment solution")
    canvas.drawRightString(7.8 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


doc = SimpleDocTemplate(
    str(out), pagesize=letter,
    rightMargin=0.7 * inch, leftMargin=0.7 * inch,
    topMargin=0.62 * inch, bottomMargin=0.68 * inch,
    title="Freight Rate Prediction - Assessment Report",
    author="Example solution",
)
story = []
story += [p("Freight Rate Prediction", "TitleCustom"), p("Machine Learning Engineer Assessment | Example solution and validation report", "SubTitle")]

selected = metrics["selected_model_by_forward_holdout_mae"]
selected_metrics = metrics["forward_holdout_metrics"][selected]
kpis = Table([
    [p(f"{metrics['development_rows']:,}", "Kpi"), p(f"{selected_metrics['mae']:,.2f}", "Kpi"), p(f"{selected_metrics['wape_percent']:.2f}%", "Kpi")],
    [p("labeled loads", "KpiLabel"), p("holdout MAE ($)", "KpiLabel"), p("holdout WAPE", "KpiLabel")],
], colWidths=[2.25 * inch] * 3, rowHeights=[0.32 * inch, 0.24 * inch])
kpis.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), light), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D3E0E3")),
    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story += [kpis, Spacer(1, 8)]

story += [p("1. Objective and data", "Section")]
story += [p("The objective is to predict <b>posted_rate</b> for each load. The supplied development data contains 48,000 labeled rows dated January through October 2025. The separate November-December file contains 12,000 unlabeled loads. The identifier <b>load_id</b> is retained only for output matching and excluded from model features.")]
story += [p("A quick audit found 300 missing weight values and 374 missing market_index values in the labeled data. The main validation file has 165 missing weights and 249 missing market_index values. Rates range from $57.22 to $25,533.00, with a median of $2,030.76. Distance has a strong raw association with posted rate (Pearson correlation about 0.91); this is a useful screening observation, not a complete explanation of rate behavior.")]

story += [p("2. Split and model selection", "Section")]
story += [p("I used a forward-in-time holdout: train on January 1-August 31 (38,477 rows) and validate on September 1-October 31 (9,523 rows). This better reflects forecasting later loads than a random split. I compared an equipment-specific median rate-per-mile baseline, a Ridge regression, and a HistGradientBoostingRegressor. Model selection used holdout MAE; RMSE and WAPE are also reported.")]
comparison = metrics["forward_holdout_metrics"]
rows = [["Model", "MAE ($)", "RMSE ($)", "WAPE", "Mean error ($)"]]
labels = {
    "equipment_median_rate_per_mile_baseline": "Median rate per mile by equipment",
    "ridge": "Ridge regression",
    "hist_gradient_boosting": "HistGradientBoosting (selected)",
}
for name, values in comparison.items():
    rows.append([labels[name], f"{values['mae']:,.2f}", f"{values['rmse']:,.2f}", f"{values['wape_percent']:.2f}%", f"{values['mean_error_prediction_minus_actual']:,.2f}"])
t = Table(rows, colWidths=[2.25 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 1.25 * inch], repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D3E0E3")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story += [t, Spacer(1, 5)]
story += [p("The selected model achieved a holdout MAE of $136.94, RMSE of $636.67, and WAPE of 5.72%. RMSE is much larger than MAE, showing that a smaller number of large misses remain; MAE is therefore the primary selection metric. The mean prediction error was -$49.31, indicating modest underprediction on this holdout.")]

story += [PageBreak(), p("Implementation and December Scenario", "TitleCustom")]
story += [p("3. Feature and missing-data handling", "Section")]
story += [p("Pickup, delivery, and equipment are one-hot encoded. Numeric inputs include coordinates, distance, weight, market index, quote signal, and calendar features derived from date (month, day of year, day of week, weekend flag, seasonal sine/cosine, and ordinal date). Numeric medians are learned only within each fitted training pipeline, and missingness indicators are included. Unseen categories are ignored safely. The model is refit on all 48,000 labeled rows after validation and then used for final predictions.")]
story += [p("The December chart input intentionally provides only pickup, delivery, distance, equipment, weight, and date. Coordinates, market index, and quote signal are unavailable in that scenario; the fitted numeric imputer supplies training medians for those absent columns. Therefore, the chart is a date-varying prediction for the supplied fixed load, not a forecast conditioned on daily market signals.")]

story += [p("4. December 2025 predicted-rate chart", "Section")]
chart_img = Image(str(chart), width=7.0 * inch, height=3.35 * inch)
chart_img.hAlign = "LEFT"
story += [chart_img, Spacer(1, 5)]
story += [p("The provided score.py accepted the 12,000-row validation file and all 31 December dates, then created this chart. The scorer validates formats, identifiers, dates, and positive predictions; it does not measure forecast accuracy. Spotter calculates the final assessment metric after submission.", "SmallCustom")]

story += [p("5. Reproduction and limitations", "Section")]
story += [p("From the project folder, install requirements, run <b>python run_assessment.py</b>, then run the scorer command shown in README.md. The script writes validation_predictions.csv, december_chart_inputs_completed.csv, and validation_metrics.json. This example uses one two-month forward holdout and limited tuning; a production-quality study would add multiple expanding-window checks and residual analysis. The measured holdout is evidence for model choice, not a guarantee of November-December accuracy.")]
story += [p("Submission note: this package is a worked example for study. Review the choices and results yourself and record your own camera-on Loom walkthrough before using any of it as an assessment response.", "SmallCustom")]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(out)
