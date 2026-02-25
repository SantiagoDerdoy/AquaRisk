# core/report_generator.py

import os
import datetime
from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


# ---------------------------------------------------------------------------
# generate_executive_pdf
# Called from app.py (Streamlit) → returns bytes for st.download_button
# ---------------------------------------------------------------------------

def generate_executive_pdf(
    well_name,
    acei_score,
    acei_category,
    exceedance_probability,
    risk_classification,
    time_to_threshold,
    final_forecast_value,
    threshold_value,
    chart_path
):
    """
    Generates an executive PDF report and returns it as bytes.

    Parameters
    ----------
    well_name               : str
    acei_score              : float
    acei_category           : str
    exceedance_probability  : float  (0–1)
    risk_classification     : str
    time_to_threshold       : int | None
    final_forecast_value    : float
    threshold_value         : float
    chart_path              : str   – path to saved forecast PNG
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    elements = []
    styles = getSampleStyleSheet()

    # -------------------------
    # Custom Styles
    # -------------------------
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor("#0B3C5D"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#1CA7A6"),
        spaceAfter=8,
        spaceBefore=16
    )

    normal_style = styles["Normal"]

    # -------------------------
    # HEADER
    # -------------------------
    elements.append(Paragraph("AquaRisk Analytics", title_style))
    elements.append(Paragraph(
        "Predictive Groundwater Intelligence Platform",
        subtitle_style
    ))

    today = datetime.date.today().strftime("%B %d, %Y")
    elements.append(Paragraph(f"<b>Well:</b> {well_name}", normal_style))
    elements.append(Paragraph(f"<b>Assessment Date:</b> {today}", normal_style))
    elements.append(Spacer(1, 0.4 * inch))

    # -------------------------
    # EXECUTIVE SUMMARY
    # -------------------------
    elements.append(Paragraph("Executive Summary", section_style))

    threshold_text = (
        f"Projected threshold crossing in {time_to_threshold} months."
        if time_to_threshold is not None
        else "Threshold not expected to be reached within forecast horizon."
    )

    summary_text = (
        f"The groundwater asset presents an ACEI™ score of <b>{round(acei_score, 2)}</b> "
        f"(<b>{acei_category}</b>), with an exceedance probability of "
        f"<b>{exceedance_probability:.1%}</b> and a risk classification of "
        f"<b>{risk_classification}</b>. "
        f"The projected final groundwater level is <b>{round(final_forecast_value, 2)} m</b> "
        f"against an operational threshold of <b>{threshold_value} m</b>. "
        f"{threshold_text}"
    )

    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------
    # KEY METRICS TABLE
    # -------------------------
    elements.append(Paragraph("Key Risk Metrics", section_style))

    data = [
        ["Indicator", "Value"],
        ["ACEI™ Score", f"{round(acei_score, 2)} / 100"],
        ["ACEI™ Category", acei_category],
        ["Exceedance Probability", f"{exceedance_probability:.1%}"],
        ["Risk Classification", risk_classification],
        ["Final Forecast Level (m)", f"{round(final_forecast_value, 2)} m"],
        ["Operational Threshold (m)", f"{threshold_value} m"],
        [
            "Time to Threshold",
            f"{time_to_threshold} months" if time_to_threshold else "Not reached"
        ],
    ]

    table = Table(data, colWidths=[3.2 * inch, 2.5 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    # -------------------------
    # CHART
    # -------------------------
    elements.append(Paragraph("Forecast Visualization", section_style))
    elements.append(Spacer(1, 0.2 * inch))

    if chart_path and os.path.exists(chart_path):
        img = Image(chart_path, width=6 * inch, height=3.5 * inch)
        elements.append(img)
    else:
        elements.append(Paragraph(
            "Chart not available.", styles["Italic"]
        ))

    # -------------------------
    # FOOTER
    # -------------------------
    elements.append(Spacer(1, 0.8 * inch))
    elements.append(Paragraph(
        "Confidential – AquaRisk Analytics | Groundwater Risk Intelligence",
        styles["Italic"]
    ))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ---------------------------------------------------------------------------
# generate_report
# Called from main.py (CLI) → writes PDF file to disk
# ---------------------------------------------------------------------------

def generate_report(
    pdf_path,
    client_name,
    location,
    report_results
):
    """
    Generates a multi-well PDF report and saves it to disk.

    Parameters
    ----------
    pdf_path       : str  – output file path
    client_name    : str
    location       : str
    report_results : list of dicts with keys:
                     well, scenarios, optimization, plot
    """

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor("#0B3C5D"),
        spaceAfter=6
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#1CA7A6"),
        spaceAfter=8,
        spaceBefore=16
    )

    normal_style = styles["Normal"]
    today = datetime.date.today().strftime("%B %d, %Y")

    elements.append(Paragraph("AquaRisk Analytics — Portfolio Report", title_style))
    elements.append(Paragraph(f"Client: {client_name}", normal_style))
    elements.append(Paragraph(f"Location: {location}", normal_style))
    elements.append(Paragraph(f"Date: {today}", normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    for well_data in report_results:

        well = well_data["well"]
        scenarios = well_data["scenarios"]
        optimization = well_data["optimization"]
        plot = well_data.get("plot")

        elements.append(Paragraph(f"Well: {well}", section_style))

        # Scenario table
        table_data = [["Scenario", "Exceedance Prob.", "Risk", "Cross (mo.)", "P(12m)", "P(24m)"]]
        for s in scenarios:
            table_data.append([
                s["name"],
                f"{s['probability']:.1%}",
                s["risk"],
                f"{s['mean_cross']:.1f}" if s["mean_cross"] is not None else "—",
                f"{s['prob_12']:.1%}" if s["prob_12"] is not None else "—",
                f"{s['prob_24']:.1%}" if s["prob_24"] is not None else "—",
            ])

        t = Table(table_data, colWidths=[1.5*inch, 1.1*inch, 1.0*inch, 0.9*inch, 0.8*inch, 0.8*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor("#f8fafc"), colors.white]),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Optimization result
        if optimization["reduction"] is not None:
            opt_text = (
                f"Pumping optimization: a <b>{optimization['reduction']*100:.0f}%</b> reduction "
                f"achieves a risk level of <b>{optimization['risk']:.1%}</b>."
            )
        else:
            opt_text = "Even with 80% pumping reduction, risk remains above the 20% target."

        elements.append(Paragraph(opt_text, normal_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Chart
        if plot and os.path.exists(plot):
            img = Image(plot, width=6 * inch, height=3.2 * inch)
            elements.append(img)

        elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(
        "Confidential – AquaRisk Analytics | Groundwater Risk Intelligence",
        styles["Italic"]
    ))

    doc.build(elements)