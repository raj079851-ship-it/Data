"""
Dashboard Builder Module
Interactive, Modular Analytics Dashboard Designer:
- 6 Professional Themes: Executive Dark, Modern Light, Cyberpunk Indigo, Minimal Slate, Emerald Corporate, Power BI Gold
- Dynamic Widget Types:
  * Executive KPI Cards
  * Trend Line / Area Charts
  * Segment Column / Bar Charts
  * Share Donut / Pie Charts
  * Interactive Pivot Data Tables
  * AI Insight & Recommendation Callouts
- Interactive Slicers & Cross-filtering
- Automated AI Dashboard Generator ("Create an executive sales dashboard")
- Layout saving and export
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

DASHBOARD_THEMES = {
    "Executive Dark": {
        "bg": "#0b0f19", "card_bg": "#1e293b", "border": "#334155",
        "text": "#f8fafc", "accent": "#6366f1", "plotly_template": "plotly_dark"
    },
    "Modern Light": {
        "bg": "#f8fafc", "card_bg": "#ffffff", "border": "#e2e8f0",
        "text": "#0f172a", "accent": "#4f46e5", "plotly_template": "plotly_white"
    },
    "Cyberpunk Indigo": {
        "bg": "#090d16", "card_bg": "#131b2e", "border": "#28354f",
        "text": "#e0e7ff", "accent": "#818cf8", "plotly_template": "plotly_dark"
    },
    "Minimal Slate": {
        "bg": "#0f172a", "card_bg": "#1e293b", "border": "#475569",
        "text": "#cbd5e1", "accent": "#38bdf8", "plotly_template": "plotly_dark"
    },
    "Emerald Corporate": {
        "bg": "#061a14", "card_bg": "#0d281e", "border": "#174634",
        "text": "#ecfdf5", "accent": "#10b981", "plotly_template": "plotly_dark"
    },
    "Power BI Gold": {
        "bg": "#181818", "card_bg": "#252526", "border": "#3c3c3d",
        "text": "#ffffff", "accent": "#f2c811", "plotly_template": "plotly_dark"
    }
}


def generate_ai_dashboard_spec(df: pd.DataFrame, intent: str = "Executive Overview") -> Dict[str, Any]:
    """
    Automatically creates an optimal dashboard layout, KPIs, and charts
    based on the user's intent and dataset schema.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    date_cols = [c for c in df.columns if any(w in c.lower() for w in ["date", "time", "created", "signup"])]

    # Select primary metric & dimension
    metric = num_cols[0] if num_cols else "Count"
    secondary_metric = num_cols[1] if len(num_cols) > 1 else metric
    dimension = cat_cols[0] if cat_cols else "Category"
    secondary_dim = cat_cols[1] if len(cat_cols) > 1 else dimension
    date_dim = date_cols[0] if date_cols else None

    widgets = [
        {
            "id": "w_kpi_1",
            "type": "kpi",
            "title": f"Total {metric}",
            "metric": metric,
            "aggregation": "sum",
            "format": "currency" if any(k in metric.lower() for k in ["price", "revenue", "charge", "cost", "profit"]) else "number"
        },
        {
            "id": "w_kpi_2",
            "type": "kpi",
            "title": f"Average {metric}",
            "metric": metric,
            "aggregation": "avg",
            "format": "number"
        },
        {
            "id": "w_kpi_3",
            "type": "kpi",
            "title": "Total Records",
            "metric": "*",
            "aggregation": "count",
            "format": "number"
        },
        {
            "id": "w_chart_bar",
            "type": "bar_chart",
            "title": f"{metric} by {dimension}",
            "x_col": dimension,
            "y_col": metric,
            "aggregation": "sum"
        },
        {
            "id": "w_chart_donut",
            "type": "donut_chart",
            "title": f"Share by {secondary_dim}",
            "names_col": secondary_dim,
            "values_col": metric
        }
    ]

    if date_dim:
        widgets.insert(3, {
            "id": "w_chart_trend",
            "type": "line_chart",
            "title": f"{metric} Trend Over Time",
            "date_col": date_dim,
            "metric_col": metric
        })

    return {
        "title": f"{intent} Dashboard",
        "theme": "Executive Dark",
        "primary_metric": metric,
        "primary_dimension": dimension,
        "widgets": widgets,
        "ai_insight": f"Dashboard automatically configured around core driver '{metric}' grouped across '{dimension}'.",
        "filter_dimension": dimension
    }
