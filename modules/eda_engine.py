"""
EDA Engine Module
Calculates descriptive statistics, distribution metrics, correlation matrices (Pearson, Spearman, Kendall),
collinearity alerts, distribution shape detection (Normality, Skewness, Kurtosis),
cross-variable relationship metrics, and automated natural-language analytical findings.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


def compute_comprehensive_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Computes in-depth statistical profiles for numerical and categorical columns."""
    # 1. Numerical statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_stats_list = []

    for col in numeric_cols:
        series = df[col]
        valid = series.dropna()
        if len(valid) == 0:
            continue
        
        q25 = float(valid.quantile(0.25))
        q75 = float(valid.quantile(0.75))
        q90 = float(valid.quantile(0.90))
        q99 = float(valid.quantile(0.99))
        iqr = q75 - q25
        mean_val = float(valid.mean())
        median_val = float(valid.median())
        std_val = float(valid.std()) if len(valid) > 1 else 0.0
        var_val = float(valid.var()) if len(valid) > 1 else 0.0
        skew_val = float(valid.skew()) if len(valid) > 2 else 0.0
        kurt_val = float(valid.kurtosis()) if len(valid) > 3 else 0.0

        # Distribution shape detection
        if abs(skew_val) < 0.5 and abs(kurt_val) < 1.0:
            shape = "Approximately Normal"
        elif skew_val > 1.0:
            shape = "Right-Skewed (Positive)"
        elif skew_val < -1.0:
            shape = "Left-Skewed (Negative)"
        elif kurt_val > 2.0:
            shape = "Heavy-Tailed (Leptokurtic)"
        else:
            shape = "Moderately Skewed"

        num_stats_list.append({
            "Feature": col,
            "Count": int(valid.count()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(series.isna().mean() * 100, 2),
            "Mean": round(mean_val, 2),
            "Std Dev": round(std_val, 2),
            "Variance": round(var_val, 2),
            "Min": round(float(valid.min()), 2),
            "25%": round(q25, 2),
            "Median (50%)": round(median_val, 2),
            "75%": round(q75, 2),
            "90%": round(q90, 2),
            "99%": round(q99, 2),
            "Max": round(float(valid.max()), 2),
            "IQR": round(float(iqr), 2),
            "Skewness": round(skew_val, 2),
            "Kurtosis": round(kurt_val, 2),
            "Distribution Shape": shape
        })

    num_df = pd.DataFrame(num_stats_list)

    # 2. Categorical statistics
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    cat_stats_list = []

    for col in cat_cols:
        series = df[col]
        valid = series.dropna()
        total_valid = len(valid)
        if total_valid == 0:
            continue
        
        val_counts = valid.value_counts()
        top_cat = str(val_counts.index[0]) if len(val_counts) > 0 else "N/A"
        top_freq = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
        top_pct = round(top_freq / total_valid * 100, 2) if total_valid > 0 else 0.0

        rare_cats = [str(k) for k, v in val_counts.items() if (v / total_valid) < 0.02]

        cat_stats_list.append({
            "Feature": col,
            "Count": int(valid.count()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(series.isna().mean() * 100, 2),
            "Unique Values": int(valid.nunique()),
            "Top Category": top_cat,
            "Top Frequency": top_freq,
            "Top Prevalence %": top_pct,
            "Rare Categories Count": len(rare_cats)
        })

    cat_df = pd.DataFrame(cat_stats_list)

    return {
        "numeric": num_df,
        "categorical": cat_df
    }


def compute_correlation_analysis(
    df: pd.DataFrame,
    method: str = "pearson", # 'pearson', 'spearman', 'kendall'
    multicollinearity_thresh: float = 0.80
) -> Dict[str, Any]:
    """Computes correlation matrix, top paired correlations, and multicollinearity alerts."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {
            "corr_matrix": pd.DataFrame(),
            "top_positive": [],
            "top_negative": [],
            "collinear_pairs": []
        }

    corr = numeric_df.corr(method=method)
    
    pairs = []
    collinear_pairs = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val):
                pair_data = {
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(val), 3),
                    "abs_corr": abs(float(val)),
                    "strength": "Strong" if abs(val) >= 0.7 else "Moderate" if abs(val) >= 0.4 else "Weak"
                }
                pairs.append(pair_data)
                if abs(float(val)) >= multicollinearity_thresh:
                    collinear_pairs.append(pair_data)

    pairs_sorted = sorted(pairs, key=lambda x: x["correlation"], reverse=True)
    top_pos = [p for p in pairs_sorted if p["correlation"] > 0][:8]
    top_neg = [p for p in reversed(pairs_sorted) if p["correlation"] < 0][:8]

    return {
        "corr_matrix": corr,
        "method": method,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "collinear_pairs": collinear_pairs
    }


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Profiles relationships between target column and predictor features."""
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataframe."}

    series = df[target_col].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series) and series.nunique() > 10

    results = {
        "target": target_col,
        "is_continuous": is_numeric,
        "unique_values": int(series.nunique()),
        "associations": []
    }

    if is_numeric:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        for col in num_cols:
            valid_df = df[[col, target_col]].dropna()
            if len(valid_df) > 5:
                r, p_val = stats.pearsonr(valid_df[col], valid_df[target_col])
                results["associations"].append({
                    "feature": col,
                    "type": "Numeric",
                    "metric": "Pearson r",
                    "score": round(float(r), 3),
                    "abs_score": abs(float(r)),
                    "p_value": round(float(p_val), 5)
                })
        results["associations"] = sorted(results["associations"], key=lambda x: x["abs_score"], reverse=True)
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            groups = [group.dropna().values for _, group in df.groupby(target_col)[col] if len(group.dropna()) > 0]
            if len(groups) > 1:
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    if not np.isnan(f_val):
                        results["associations"].append({
                            "feature": col,
                            "type": "Numeric vs Category",
                            "metric": "ANOVA F-Score",
                            "score": round(float(f_val), 2),
                            "abs_score": round(float(f_val), 2),
                            "p_value": round(float(p_val), 5)
                        })
                except Exception:
                    pass
        results["associations"] = sorted(results["associations"], key=lambda x: x["score"], reverse=True)

    return results


def generate_automated_eda_insights(df: pd.DataFrame) -> List[str]:
    """Generates natural language automated analytical findings."""
    insights = []
    n_rows, n_cols = df.shape
    insights.append(f"Dataset contains {n_rows:,} records across {n_cols} attributes.")

    # Check top correlations
    corr_info = compute_correlation_analysis(df, method="pearson")
    if corr_info["top_positive"]:
        p = corr_info["top_positive"][0]
        insights.append(f"Strongest positive relationship found between '{p['feature_1']}' and '{p['feature_2']}' (r = {p['correlation']}).")
    if corr_info["collinear_pairs"]:
        insights.append(f"Warning: {len(corr_info['collinear_pairs'])} variable pair(s) exhibit high collinearity (|r| ≥ 0.80), risking variance inflation in linear models.")

    # Numerical distributions
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:4]:
        s = df[col].dropna()
        if len(s) > 10:
            skew = s.skew()
            if abs(skew) > 1.5:
                insights.append(f"Attribute '{col}' has severe skewness ({round(skew, 2)}), with values ranging from {round(s.min(), 1)} to {round(s.max(), 1)}.")

    # High cardinality
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols[:3]:
        nu = df[col].nunique()
        if nu == len(df):
            insights.append(f"Column '{col}' is completely unique for every row, functioning as an identifier key.")

    return insights
