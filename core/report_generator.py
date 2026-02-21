# report_generator.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


def generate_report(filepath, client, location, report_results):

    doc = SimpleDocTemplate(filepath, pagesize=pagesizes.A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    normal_style = styles["Normal"]

    elements.append(Paragraph("AquaRisk Predictive Intelligence Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Client: {client}", normal_style))
    elements.append(Paragraph(f"Location: {location}", normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    for well_data in report_results:

        elements.append(Paragraph(f"Well: {well_data['well']}", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        table_data = [["Scenario", "Risk", "Prob.", "Mean Cross (m)", "≤12m", "≤24m"]]

        for scenario in well_data["scenarios"]:
            table_data.append([
                scenario["name"],
                scenario["risk"],
                f"{scenario['probability']:.2%}",
                "-" if scenario["mean_cross"] is None else f"{scenario['mean_cross']:.1f}",
                f"{scenario['prob_12']:.2%}",
                f"{scenario['prob_24']:.2%}"
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (2, 1), (-1, -1), 'CENTER')
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Image(well_data["plot"], width=6 * inch, height=3.5 * inch))
        elements.append(Spacer(1, 0.6 * inch))

    doc.build(elements)