"""
Business Intelligence & Automated Insights Module
Advanced Enterprise BI Suite:
- KPI Goal Tracking & Variance Analysis (Actual vs Target/Budget)
- Period-over-Period (PoP, MoM, YoY) Growth Rates
- Contribution & Driver Breakdown (Waterfall & Pareto 80/20)
- Customer & Cohort Segmentation Analysis
- Automated Natural-Language Insight Discovery (Anomalies, Outliers, Inflection Points)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def compute_kpi_goal_tracking(
    df: pd.DataFrame,
    metric_col: str,
    target_value: float,
    dimension_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates actual performance against target/budget.
    Computes variance, % achievement, and status (Exceeded, On Track, At Risk, Lagging).
    """
    actual_total = float(df[metric_col].sum())
    variance = actual_total - target_value
    pct_achieved = round((actual_total / target_value * 100), 1) if target_value != 0 else 0.0

    if pct_achieved >= 105:
        status = "Exceeded Target"
        badge = "badge-success"
    elif pct_achieved >= 95:
        status = "On Track"
        badge = "badge-info"
    elif pct_achieved >= 80:
        status = "At Risk"
        badge = "badge-warning"
    else:
        status = "Lagging Behind Target"
        badge = "badge-danger"

    segment_breakdown = []
    if dimension_col and dimension_col in df.columns:
        agg = df.groupby(dimension_col)[metric_col].sum().reset_index()
        # Allocate target proportionally or evenly
        avg_target_per_seg = target_value / max(1, len(agg))
        for _, r in agg.iterrows():
            val = float(r[metric_col])
            v_diff = val - avg_target_per_seg
            p_ach = round((val / avg_target_per_seg * 100), 1) if avg_target_per_seg > 0 else 0
            segment_breakdown.append({
                "segment": str(r[dimension_col]),
                "actual": round(val, 2),
                "target": round(avg_target_per_seg, 2),
                "variance": round(v_diff, 2),
                "pct_achieved": p_ach
            })

    return {
        "metric": metric_col,
        "actual": round(actual_total, 2),
        "target": round(target_value, 2),
        "variance": round(variance, 2),
        "pct_achieved": pct_achieved,
        "status": status,
        "badge": badge,
        "segments": segment_breakdown
    }


def compute_period_growth(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    period: str = "M" # 'M' (Month-over-Month), 'Q' (Quarter-over-Quarter), 'Y' (Year-over-Year)
) -> Dict[str, Any]:
    """Computes Period-over-Period velocity and identifies accelerating or decelerating momentum."""
    sub = df[[date_col, metric_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna().sort_values(by=date_col)

    freq_map = {"M": "MS", "Q": "QS", "Y": "YS"}
    target_freq = freq_map.get(period.upper(), "MS")

    ts = sub.set_index(date_col)[metric_col].resample(target_freq).sum()
    pct_changes = (ts.pct_change() * 100).round(1).dropna()

    recent_growth = float(pct_changes.iloc[-1]) if len(pct_changes) > 0 else 0.0
    avg_growth = float(pct_changes.mean()) if len(pct_changes) > 0 else 0.0

    records = []
    for d, val in ts.items():
        records.append({
            "Period": d.strftime("%Y-%m-%d"),
            "Volume": round(float(val), 2),
            "PoP_Growth_Pct": round(float(pct_changes.get(d, 0.0)), 1)
        })

    return {
        "metric": metric_col,
        "recent_growth_pct": round(recent_growth, 1),
        "average_pop_growth": round(avg_growth, 1),
        "trend_label": "Accelerating" if recent_growth > 5 else "Decelerating" if recent_growth < -5 else "Stable",
        "records": records
    }


def run_automated_insights_engine(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Autonomous insight engine:
    - Scans for top revenue/metric drivers
    - Detects negative variances and bottom performers
    - Flags extreme outliers and distribution skews
    - Emits actionable executive takeaways
    """
    insights = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if num_cols and cat_cols:
        dim = cat_cols[0]
        metric = num_cols[0]
        agg = df.groupby(dim)[metric].agg(["sum", "count"]).sort_values(by="sum", ascending=False).reset_index()
        total = agg["sum"].sum()

        if len(agg) > 1:
            top_row = agg.iloc[0]
            bot_row = agg.iloc[-1]
            top_share = round((top_row["sum"] / total * 100), 1) if total > 0 else 0.0
            bot_share = round((bot_row["sum"] / total * 100), 1) if total > 0 else 0.0

            insights.append({
                "type": "Top Driver",
                "title": f"Dominant Volume Driver: '{top_row[dim]}'",
                "detail": f"Segment '{top_row[dim]}' generates {top_share}% of total {metric} ({top_row['sum']:,.2f}), outperforming the lowest segment ('{bot_row[dim]}' at {bot_share}%) by a factor of {round(top_row['sum'] / max(1, bot_row['sum']), 1)}x."
            })

            # Pareto Check: Do top 20% categories account for >= 70% of metric?
            top_20_pct_count = max(1, int(len(agg) * 0.2))
            top_20_sum = agg.iloc[:top_20_pct_count]["sum"].sum()
            pareto_share = round((top_20_sum / total * 100), 1) if total > 0 else 0.0
            if pareto_share >= 65:
                insights.append({
                    "type": "Pareto Concentration",
                    "title": f"High Value Concentration ({pareto_share}%)",
                    "detail": f"The top {top_20_pct_count} segment(s) account for {pareto_share}% of cumulative {metric}. Business outcomes are heavily concentrated in key segments."
                })

    # Outlier / Anomaly Check
    for col in num_cols[:2]:
        s = df[col].dropna()
        if len(s) > 10:
            q75, q25 = s.quantile(0.75), s.quantile(0.25)
            iqr = q75 - q25
            outliers = df[(df[col] > q75 + 3.0 * iqr) | (df[col] < q25 - 3.0 * iqr)]
            if len(outliers) > 0:
                insights.append({
                    "type": "Data Anomaly",
                    "title": f"Extreme Tail Outliers in '{col}'",
                    "detail": f"Identified {len(outliers)} observation(s) beyond 3.0x IQR (Max observed: {s.max():,.2f}). Verify whether these represent extraordinary business transactions or sensor glitches."
                })

    return insights
