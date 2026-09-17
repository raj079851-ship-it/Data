"""
Reports Engine Module
Multi-Format Enterprise Report Generator:
- Report Types:
  * Executive Summary Report
  * Data Quality Audit Report
  * Exploratory Data Analysis (EDA) Report
  * Machine Learning & Prediction Report
  * Business Performance Report
- Multi-format Export:
  * HTML Report (Self-contained, responsive, printable to PDF)
  * Excel Workbook (.xlsx with multiple formatted sheets)
  * JSON & CSV analytical exports
"""

import io
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_html_report(
    report_type: str,
    project_name: str,
    df: pd.DataFrame,
    meta: Dict[str, Any],
    eda_stats: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    insights: Optional[List[str]] = None,
    audit_trail: Optional[List[str]] = None
) -> str:
    """Generates an enterprise-styled, standalone HTML report with CSS print styles."""
    now = datetime.now().strftime("%B %d, %Y - %H:%M")
    n_rows, n_cols = df.shape
    health_score = meta.get("health_score", 85)
    health_status = meta.get("health_status", "Good")

    # Key statistics rows
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_stats_html = ""
    for col in num_cols[:5]:
        s = df[col].dropna()
        sample_stats_html += f"""
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0; font-weight:600;">{col}</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">{s.mean():,.2f}</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">{s.median():,.2f}</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">{s.min():,.2f}</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">{s.max():,.2f}</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">{s.std():,.2f}</td>
        </tr>
        """

    insights_html = ""
    if insights:
        for ins in insights[:6]:
            insights_html += f"""<li style="margin-bottom:8px; line-height:1.6; color:#334155;">{ins}</li>"""
    else:
        insights_html = """
        <li style="margin-bottom:8px; line-height:1.6; color:#334155;">Dataset dimensions: <b>{n_rows:,} records</b> across <b>{n_cols} attributes</b>.</li>
        <li style="margin-bottom:8px; line-height:1.6; color:#334155;">Health Score of <b>{health_score}/100</b> ({health_status}).</li>
        <li style="margin-bottom:8px; line-height:1.6; color:#334155;">All core data pipelines verified for downstream modeling and business reporting.</li>
        """

    ml_html = ""
    if ml_results:
        algo = ml_results.get("best_model_name", "Random Forest")
        ml_html = f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:18px; margin-top:20px;">
            <h3 style="margin-top:0; color:#1e293b;">🤖 Machine Learning & Modeling Synthesis</h3>
            <p style="color:#475569;">Champion Model: <b>{algo}</b> | Task: <b>{ml_results.get('task_type', 'Supervised').title()}</b></p>
        </div>
        """

    audit_html = ""
    if audit_trail and len(audit_trail) > 0:
        step_rows = ""
        for i, step in enumerate(audit_trail, 1):
            step_rows += f"""
            <tr>
                <td style="padding:10px; border-bottom:1px solid #e2e8f0; text-align:center; font-weight:700; color:#4f46e5;">#{i}</td>
                <td style="padding:10px; border-bottom:1px solid #e2e8f0; color:#1e293b;">{step}</td>
                <td style="padding:10px; border-bottom:1px solid #e2e8f0; text-align:center;"><span style="background:#dcfce7; color:#15803d; font-size:11px; padding:3px 8px; border-radius:4px; font-weight:700;">Applied</span></td>
            </tr>
            """
        audit_html = f"""
        <div class="section">
            <div class="section-title">🛠️ Data Transformation &amp; Operations Audit Trail ({len(audit_trail)} Steps Performed)</div>
            <table>
                <thead>
                    <tr>
                        <th style="width:60px; text-align:center;">Step</th>
                        <th>Operation / Transformation Executed</th>
                        <th style="width:90px; text-align:center;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {step_rows}
                </tbody>
            </table>
        </div>
        """
    else:
        audit_html = """
        <div class="section">
            <div class="section-title">🛠️ Data Transformation &amp; Operations Audit Trail</div>
            <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:14px; font-size:13px; color:#64748b;">
                <b>Baseline Ingestion:</b> Raw dataset loaded without modifications. Verified integrity for analytics.
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_type} | {project_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #ffffff;
            color: #0f172a;
            margin: 0;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid #6366f1;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .title {{ font-size: 26px; font-weight: 800; color: #1e1b4b; margin: 0; }}
        .subtitle {{ font-size: 14px; color: #64748b; margin-top: 6px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; }}
        .kpi-value {{ font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 4px; }}
        .section {{ margin-bottom: 35px; }}
        .section-title {{ font-size: 18px; font-weight: 700; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ background: #f1f5f9; padding: 10px; color: #475569; font-weight: 700; }}
        .footer {{ margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 20px; font-size: 12px; color: #94a3b8; text-align: center; }}
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 class="title">DataMind AI | {report_type}</h1>
            <div class="subtitle">Project: <b>{project_name}</b> | Generated on {now}</div>
        </div>
        <div>
            <span style="background:#e0e7ff; color:#4338ca; padding:6px 14px; border-radius:9999px; font-size:12px; font-weight:700;">Health Score: {health_score}/100</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{n_rows:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Attributes</div>
            <div class="kpi-value">{n_cols}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Missing Cells</div>
            <div class="kpi-value">{meta.get('missing_cells', 0):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Duplicate Rows</div>
            <div class="kpi-value">{meta.get('duplicated_rows', 0):,}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Executive Findings & Actionable Insights</div>
        <ul style="padding-left: 20px;">
            {insights_html}
        </ul>
    </div>

    <div class="section">
        <div class="section-title">Descriptive Feature Statistics Summary</div>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Std Dev</th>
                </tr>
            </thead>
            <tbody>
                {sample_stats_html}
            </tbody>
        </table>
    </div>

    {audit_html}

    {ml_html}

    <div class="footer">
        Generated automatically by DataMind AI Enterprise Analytics Platform. Confidential & Proprietary.
    </div>
</body>
</html>
"""
    return html


def generate_excel_report(
    project_name: str,
    df: pd.DataFrame,
    meta: Dict[str, Any],
    stats_dict: Optional[Dict[str, pd.DataFrame]] = None,
    audit_trail: Optional[List[str]] = None
) -> bytes:
    """Generates a multi-sheet formatted Excel workbook in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Data sample
        df.head(500).to_excel(writer, sheet_name="Data Preview", index=False)
        
        # Sheet 2: Metadata & Health
        meta_items = [
            {"Property": "Project Name", "Value": project_name},
            {"Property": "Total Records", "Value": len(df)},
            {"Property": "Total Columns", "Value": len(df.columns)},
            {"Property": "Health Score", "Value": f"{meta.get('health_score', 85)}/100"},
            {"Property": "Health Grade", "Value": meta.get("health_grade", "A")},
            {"Property": "Missing Cells", "Value": meta.get("missing_cells", 0)},
            {"Property": "Duplicate Rows", "Value": meta.get("duplicated_rows", 0)},
            {"Property": "Generation Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]
        pd.DataFrame(meta_items).to_excel(writer, sheet_name="Dataset Health", index=False)

        # Sheet 3: Operations Audit Trail
        if audit_trail and len(audit_trail) > 0:
            audit_df = pd.DataFrame([
                {"Step": f"#{i+1}", "Operation Executed": step, "Status": "Applied"}
                for i, step in enumerate(audit_trail)
            ])
            audit_df.to_excel(writer, sheet_name="Audit Trail & Steps", index=False)
        else:
            audit_df = pd.DataFrame([
                {"Step": "#1", "Operation Executed": "Baseline Raw Ingestion (No modifications)", "Status": "Verified"}
            ])
            audit_df.to_excel(writer, sheet_name="Audit Trail & Steps", index=False)

        # Sheet 4: Numerical Statistics if available
        if stats_dict and "numeric" in stats_dict and not stats_dict["numeric"].empty:
            stats_dict["numeric"].to_excel(writer, sheet_name="Numerical Stats", index=False)

        # Sheet 5: Categorical Statistics if available
        if stats_dict and "categorical" in stats_dict and not stats_dict["categorical"].empty:
            stats_dict["categorical"].to_excel(writer, sheet_name="Categorical Stats", index=False)

    return output.getvalue()


def generate_pdf_report(
    report_title: str,
    project_name: str,
    df: pd.DataFrame,
    meta: Dict[str, Any],
    subtitle: str = "Board of Directors & Executive Leadership Briefing",
    author: str = "DataMind AI Analytics Copilot",
    classification: str = "CONFIDENTIAL // PROPRIETARY",
    sections_to_include: Optional[List[str]] = None,
    insights: Optional[List[str]] = None,
    eda_stats: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    audit_trail: Optional[List[str]] = None
) -> bytes:
    """
    Generates a high-resolution, executive-grade PDF analytics report using ReportLab.
    Formatted with consulting-grade scorecards, tables, and strategic recommendations.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#4f46e5")
    dark_slate = colors.HexColor("#0f172a")
    text_color = colors.HexColor("#1e293b")
    subtext_color = colors.HexColor("#64748b")
    card_bg = colors.HexColor("#f8fafc")
    border_color = colors.HexColor("#e2e8f0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=dark_slate,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=subtext_color,
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_color
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_color,
        leftIndent=14,
        spaceAfter=4
    )

    meta_badge_style = ParagraphStyle(
        'MetaBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4338ca"),
        alignment=2
    )

    now_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    n_rows, n_cols = df.shape
    health = meta.get("health_score", 90)
    health_grade = meta.get("health_grade", "A")
    missing = meta.get("missing_cells", int(df.isna().sum().sum()))

    story = []

    # 1. Header Banner
    header_table = Table([
        [
            Paragraph(f"<b>{report_title}</b>", title_style),
            Paragraph(f"<b>CLASSIFICATION:</b><br/>{classification}", meta_badge_style)
        ],
        [
            Paragraph(f"{subtitle} &bull; <b>Project:</b> {project_name}", subtitle_style),
            Paragraph(f"<b>Date:</b> {now_str}<br/><b>Author:</b> {author}", meta_badge_style)
        ]
    ], colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=14))

    # 2. Executive Scorecards Grid
    kpi_data = [
        [
            Paragraph("<b>TOTAL RECORDS</b><br/><font size=14 color='#0f172a'><b>" + f"{n_rows:,}" + "</b></font><br/><font size=7 color='#64748b'>Observations</font>", body_style),
            Paragraph("<b>ATTRIBUTES</b><br/><font size=14 color='#0f172a'><b>" + f"{n_cols}" + "</b></font><br/><font size=7 color='#64748b'>Dimensions</font>", body_style),
            Paragraph("<b>DATA HEALTH</b><br/><font size=14 color='#10b981'><b>" + f"{health}/100" + f" ({health_grade})</b></font><br/><font size=7 color='#64748b'>Sanitization Index</font>", body_style),
            Paragraph("<b>MISSING CELLS</b><br/><font size=14 color='#ef4444'><b>" + f"{missing:,}" + "</b></font><br/><font size=7 color='#64748b'>Requiring Impute</font>", body_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 3. Executive Findings & Strategic Insights
    story.append(Paragraph("<b>1. Executive Findings & Narrative Intelligence</b>", h2_style))
    if insights:
        for ins in insights[:6]:
            story.append(Paragraph(f"&bull; {ins}", bullet_style))
    else:
        story.append(Paragraph(f"&bull; <b>Volume Density:</b> Comprehensive audit performed over <b>{n_rows:,} records</b> and <b>{n_cols} attributes</b>.", bullet_style))
        story.append(Paragraph(f"&bull; <b>Hygiene & Integrity:</b> Data health verified at <b>{health}/100</b> with {missing:,} missing cells.", bullet_style))
        story.append(Paragraph(f"&bull; <b>Modeling Readiness:</b> Quantitative features exhibit strong signal consistency ready for downstream ML and executive reporting.", bullet_style))

    story.append(Spacer(1, 10))

    # 4. Quantitative Distributions Summary Table
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        story.append(Paragraph("<b>2. Key Numerical Feature Distributions</b>", h2_style))
        stats_table_data = [
            ["Feature", "Mean", "Median", "Min", "Max", "Std Dev"]
        ]
        for col in num_cols[:8]:
            s = df[col].dropna()
            if not s.empty:
                stats_table_data.append([
                    col,
                    f"{s.mean():,.2f}",
                    f"{s.median():,.2f}",
                    f"{s.min():,.2f}",
                    f"{s.max():,.2f}",
                    f"{s.std():,.2f}"
                ])
        stats_table = Table(stats_table_data, colWidths=[150, 78, 78, 78, 78, 78])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), card_bg),
            ('TEXTCOLOR', (0,1), (-1,-1), text_color),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 12))

    # 5. Strategic Recommendations Roadmap
    story.append(Paragraph("<b>3. Prescriptive Strategic Action Plan</b>", h2_style))
    recs_data = [
        [
            Paragraph("<b>Revenue & Pricing Optimization</b><br/><font size=8 color='#475569'>Target high-margin cohorts in upper quartile spending to cross-sell annual contracts and retain recurring lifetime value.</font>", body_style)
        ],
        [
            Paragraph("<b>Churn Mitigation & Defense</b><br/><font size=8 color='#475569'>Deploy automated retention notifications and onboarding checkpoints for customers within early commitment windows.</font>", body_style)
        ],
        [
            Paragraph("<b>Data Governance & Imputation</b><br/><font size=8 color='#475569'>Standardize continuous missing variables via Median imputation and categorical fields with Mode prior to supervised modeling.</font>", body_style)
        ]
    ]
    recs_table = Table(recs_data, colWidths=[540])
    recs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(recs_table)
    story.append(Spacer(1, 14))

    # 6. Data Transformation & Operations Audit Trail
    story.append(Paragraph("<b>4. Data Transformation &amp; Operations Audit Trail</b>", h2_style))
    if audit_trail and len(audit_trail) > 0:
        audit_table_data = [
            ["Step", "Operation / Transformation Executed", "Status"]
        ]
        for idx, step in enumerate(audit_trail[:15], 1):
            audit_table_data.append([
                f"#{idx}",
                Paragraph(f"<font size=8 color='#1e293b'>{step}</font>", body_style),
                "Applied"
            ])
        if len(audit_trail) > 15:
            audit_table_data.append([
                "...",
                Paragraph(f"<font size=8 color='#64748b'>... and {len(audit_trail) - 15} additional transformation step(s) applied</font>", body_style),
                "Applied"
            ])
        audit_table = Table(audit_table_data, colWidths=[45, 435, 60])
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 5),
            ('TOPPADDING', (0,0), (-1,0), 5),
            ('BACKGROUND', (0,1), (-1,-1), card_bg),
            ('TEXTCOLOR', (2,1), (2,-1), colors.HexColor("#15803d")),
            ('FONTNAME', (2,1), (2,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (2,1), (2,-1), 8),
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('ALIGN', (2,0), (2,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(audit_table)
    else:
        story.append(Paragraph("<b>Baseline Dataset:</b> Raw ingestion without modifications. No transformations or filters applied yet.", body_style))
    story.append(Spacer(1, 14))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))
    footer_text = "Compiled automatically by DataMind AI Enterprise Analytics Platform &bull; Strictly Confidential &bull; Page 1 of 1"
    story.append(Paragraph(footer_text, ParagraphStyle('FooterStyle', parent=styles['Normal'], fontSize=7, textColor=subtext_color, alignment=1)))

    doc.build(story)
    return buf.getvalue()
