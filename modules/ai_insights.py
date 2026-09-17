"""
AI Insights Module
Generates automated narrative insights, executive summaries, and recommendations
using a dual-engine architecture:
1. Zero-dependency Built-in Heuristic AI Engine (Works offline, instantly, no API key needed)
2. Optional LLM Engine (Google Gemini) for deep strategic queries and conversational data Q&A
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def generate_heuristic_insights(
    df: pd.DataFrame,
    audit_data: Dict[str, Any],
    stats_data: Dict[str, pd.DataFrame],
    corr_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates rich, contextual natural language insights and recommendations
    based on comprehensive statistical and structural signals.
    """
    rows = len(df)
    cols = len(df.columns)
    
    # 1. Executive Summary
    health_score = audit_data.get("health_score", 100)
    missing_cells = audit_data.get("missing_cells", 0)
    dup_count = audit_data.get("duplicate_rows", 0)

    if health_score >= 85:
        health_status = "Excellent"
        health_desc = "The dataset is structurally sound with minimal data hygiene issues."
    elif health_score >= 65:
        health_status = "Fair / Moderate"
        health_desc = "Noticeable missing values or outlier anomalies require attention before modeling."
    else:
        health_status = "Poor / High Risk"
        health_desc = "Significant data quality defects (heavy missingness, duplicates, or extreme outliers) detected."

    exec_summary = (
        f"Analyzed **{rows:,} rows** and **{cols} features**. Overall dataset health is rated **{health_score}/100 ({health_status})**. "
        f"{health_desc}"
    )

    # 2. Key Data Quality Signals
    quality_signals = []
    if dup_count > 0:
        quality_signals.append(
            f"**Redundant Rows**: {dup_count} duplicate record(s) ({round(dup_count / rows * 100, 1)}%) identified. Recommend deduplication."
        )
    if missing_cells > 0:
        cols_with_miss = audit_data.get("cols_with_missing_count", 0)
        quality_signals.append(
            f"**Incomplete Records**: {missing_cells:,} missing data cells spread across {cols_with_miss} column(s). Imputation or filtering recommended."
        )
    constant_cols = audit_data.get("constant_cols", [])
    if constant_cols:
        quality_signals.append(
            f"**Zero-Variance Attributes**: Columns `{', '.join(constant_cols)}` carry single unique values and contribute zero predictive information."
        )
    outlier_summary = audit_data.get("outlier_summary", {})
    if outlier_summary:
        outlier_cols = [f"`{col}` ({info['iqr_outliers']} outliers)" for col, info in outlier_summary.items()]
        quality_signals.append(
            f"**Outlier Concentrations**: Detected extreme tail observations in {', '.join(outlier_cols[:4])}."
        )

    if not quality_signals:
        quality_signals.append("No critical data quality risks detected. Dataset has complete entries and balanced distributions.")

    # 3. Statistical & Distribution Observations
    stat_signals = []
    num_df = stats_data.get("numeric", pd.DataFrame())
    if not num_df.empty and "Skewness" in num_df.columns:
        # Check high skewness
        skewed = num_df[num_df["Skewness"].abs() > 1.2]
        if not skewed.empty:
            skewed_names = skewed["Feature"].tolist()
            stat_signals.append(
                f"**Distribution Asymmetry**: Significant skewness observed in `{', '.join(skewed_names[:4])}`. Applying log1p or power transformation will normalize these distributions."
            )
        # Check high variance
        high_spread = num_df.sort_values(by="Std Dev", ascending=False).head(2)
        if not high_spread.empty:
            lead_col = high_spread.iloc[0]["Feature"]
            lead_std = high_spread.iloc[0]["Std Dev"]
            stat_signals.append(
                f"**High Dispersion**: Feature `{lead_col}` demonstrates high variance (Std Dev: {lead_std:,}). Feature scaling (Z-score/MinMax) is recommended."
            )

    cat_df = stats_data.get("categorical", pd.DataFrame())
    if not cat_df.empty and "Top Prevalence %" in cat_df.columns:
        dominant_cats = cat_df[cat_df["Top Prevalence %"] > 75]
        if not dominant_cats.empty:
            dom_col = dominant_cats.iloc[0]["Feature"]
            dom_val = dominant_cats.iloc[0]["Top Category"]
            dom_pct = dominant_cats.iloc[0]["Top Prevalence %"]
            stat_signals.append(
                f"**Class Imbalance / Dominance**: Column `{dom_col}` is heavily dominated by category `'{dom_val}'` ({dom_pct}% prevalence)."
            )

    # 4. Correlation & Collinearity Insights
    corr_signals = []
    top_pos = corr_data.get("top_positive", [])
    top_neg = corr_data.get("top_negative", [])
    collinear = corr_data.get("collinear_pairs", [])

    if collinear:
        collinear_text = [f"`{p['feature_1']}` & `{p['feature_2']}` (r = {p['correlation']})" for p in collinear]
        corr_signals.append(
            f"**Collinearity Alert**: Strong collinearity detected between {', '.join(collinear_text)}. One of the redundant features can be dropped to prevent multicollinearity in linear models."
        )

    if top_pos:
        p = top_pos[0]
        corr_signals.append(
            f"**Strong Positive Association**: Highest positive correlation between `{p['feature_1']}` and `{p['feature_2']}` (r = {p['correlation']})."
        )

    if top_neg:
        p = top_neg[0]
        corr_signals.append(
            f"**Inverse Association**: Strongest negative correlation between `{p['feature_1']}` and `{p['feature_2']}` (r = {p['correlation']})."
        )

    if not corr_signals:
        corr_signals.append("Numerical features exhibit moderate to weak pairwise linear correlations.")

    # 5. Strategic Next Steps & ML Recommendations
    action_items = [
        "**Cleaning Strategy**: Execute 1-Click Auto-Clean or targeted median imputation for missing values and duplicate pruning.",
        "**Feature Engineering**: Standardize high-variance numerical columns and one-hot encode low-cardinality categorical variables.",
        "**Modeling Readiness**: If training ML models, inspect correlated feature pairs to prevent overfitting and test interaction ratios."
    ]

    return {
        "exec_summary": exec_summary,
        "health_score": health_score,
        "health_status": health_status,
        "quality_signals": quality_signals,
        "stat_signals": stat_signals,
        "corr_signals": corr_signals,
        "action_items": action_items
    }


def query_gemini_insights(
    user_query: str,
    df_context_summary: str,
    api_key: str
) -> str:
    """
    Optional LLM integration using Google Gemini SDK or HTTP request.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
You are an expert Chief Data Scientist and Analytics Advisor.
Here is the summary profile and statistical context of the user's uploaded dataset:
{df_context_summary}

User Question/Request:
{user_query}

Provide a clear, structured, and highly insightful response with actionable analytical recommendations.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Could not generate Gemini response: {e}"
