"""
Experimentation & A/B Testing Module
Provides end-to-end hypothesis evaluation for online experiments:
- Conversion Comparison (Two-proportion pooled Z-test, binary outcomes)
- Continuous Mean Comparison (Welch's independent two-sample t-test)
- Confidence Intervals (for difference in proportions & difference in means)
- Effect Size metrics (Cohen's h for conversions, Cohen's d for continuous metrics)
- Post-hoc Statistical Power & Pre-experiment Sample Size Calculator (for 80%/90% power)
- Plain-language executive conclusions and rollout recommendations
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def run_ab_conversion_test(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    control_val: Any,
    treatment_val: Any,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Executes a two-proportion Z-test for binary conversion metrics.
    Compares Control vs Treatment cohorts.
    """
    if group_col not in df.columns or target_col not in df.columns:
        return {"error": f"Columns '{group_col}' or '{target_col}' not found in dataset."}

    sub = df[[group_col, target_col]].dropna()
    ctrl_rows = sub[sub[group_col].astype(str) == str(control_val)]
    treat_rows = sub[sub[group_col].astype(str) == str(treatment_val)]

    n_c = len(ctrl_rows)
    n_t = len(treat_rows)

    if n_c < 10 or n_t < 10:
        return {"error": f"Insufficient sample size (Control: {n_c}, Treatment: {n_t}). Minimum 10 observations required per variant."}

    # Binary conversion helper
    def parse_binary(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.apply(lambda x: 1 if float(x) > 0 else 0)
        return series.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "converted", "success"]).astype(int)

    c_binary = parse_binary(ctrl_rows[target_col])
    t_binary = parse_binary(treat_rows[target_col])

    x_c = int(c_binary.sum())
    x_t = int(t_binary.sum())

    p_c = x_c / n_c
    p_t = x_t / n_t

    # Difference & Lift
    diff = p_t - p_c
    rel_lift = (diff / p_c * 100.0) if p_c > 0 else 0.0

    # Pooled proportion for standard error under null hypothesis
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t)) if 0 < p_pool < 1 else 0.00001
    z_score = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
    p_value = max(0.000001, min(1.0, float(p_value)))

    # Confidence Interval of the Difference (unpooled standard error)
    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se_unpooled = math.sqrt((p_c * (1.0 - p_c) / n_c) + (p_t * (1.0 - p_t) / n_t)) if (p_c > 0 or p_t > 0) else 0.00001
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    # Effect Size: Cohen's h = 2*arcsin(sqrt(p_t)) - 2*arcsin(sqrt(p_c))
    phi_t = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p_t))))
    phi_c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p_c))))
    cohen_h = abs(phi_t - phi_c)
    h_label = "Small" if cohen_h < 0.2 else ("Medium" if cohen_h < 0.5 else "Large")

    # Relative Risk & Odds Ratio
    rel_risk = (p_t / p_c) if p_c > 0 else 1.0
    odds_c = (p_c / (1.0 - p_c)) if p_c < 1 else 999.0
    odds_t = (p_t / (1.0 - p_t)) if p_t < 1 else 999.0
    odds_ratio = (odds_t / odds_c) if odds_c > 0 else 1.0

    # Post-hoc Power Calculation
    power_est = float(stats.norm.cdf(abs(z_score) - z_crit)) if z_crit > 0 else 0.5
    power_est = max(0.05, min(0.999, power_est))

    # Required sample size per variant for 80% power at alpha=0.05
    z_beta_80 = 0.8416
    if abs(diff) > 0.0001:
        req_n_per_variant = int(math.ceil(2.0 * ((z_crit + z_beta_80) ** 2) * p_pool * (1.0 - p_pool) / (diff ** 2)))
    else:
        req_n_per_variant = 50000

    # Significance determination
    is_significant = bool(p_value <= alpha)
    if is_significant:
        if diff > 0:
            winner = "Treatment (Variant)"
            status_tag = "Significant Winner"
            conclusion = (
                f"Treatment achieved a statistically significant conversion lift of +{rel_lift:.2f}% "
                f"({p_c*100:.2f}% vs {p_t*100:.2f}%, p = {p_value:.4f} < {alpha:.2f}). "
                f"We are {int(confidence_level*100)}% confident that the true conversion improvement lies between "
                f"+{ci_low*100:.2f}% and +{ci_high*100:.2f}%. Recommendation: Deploy Treatment to 100% traffic."
            )
        else:
            winner = "Control (Baseline)"
            status_tag = "Significant Underperformer"
            conclusion = (
                f"Treatment resulted in a statistically significant conversion drop of {rel_lift:.2f}% "
                f"({p_t*100:.2f}% vs {p_c*100:.2f}%, p = {p_value:.4f} < {alpha:.2f}). "
                f"Recommendation: Reject Treatment and maintain Control."
            )
    else:
        winner = "Inconclusive / No Winner"
        status_tag = "Not Statistically Significant"
        conclusion = (
            f"Observed conversion difference ({diff*100:+.2f}%, lift: {rel_lift:+.2f}%) is not statistically significant "
            f"at the {int(confidence_level*100)}% confidence level (p = {p_value:.4f} >= {alpha:.2f}). "
            f"Observed variances can be explained by random sampling fluctuation. "
            f"Observed statistical power is {power_est*100:.1f}%. To detect this effect with 80% confidence, run the test until collecting ~{req_n_per_variant:,} observations per variant."
        )

    return {
        "test_type": "conversion",
        "metric_name": target_col,
        "group_col": group_col,
        "confidence_level": confidence_level,
        "control_name": str(control_val),
        "treatment_name": str(treatment_val),
        "control": {
            "n": n_c,
            "conversions": x_c,
            "rate": round(p_c, 5),
            "rate_pct": round(p_c * 100.0, 2)
        },
        "treatment": {
            "n": n_t,
            "conversions": x_t,
            "rate": round(p_t, 5),
            "rate_pct": round(p_t * 100.0, 2)
        },
        "difference": round(diff, 5),
        "difference_pct": round(diff * 100.0, 3),
        "relative_lift_pct": round(rel_lift, 2),
        "z_statistic": round(z_score, 4),
        "p_value": round(p_value, 6),
        "confidence_interval": {
            "low": round(ci_low, 5),
            "high": round(ci_high, 5),
            "low_pct": round(ci_low * 100.0, 3),
            "high_pct": round(ci_high * 100.0, 3)
        },
        "effect_size": {
            "metric": "Cohen's h",
            "value": round(cohen_h, 4),
            "magnitude": h_label,
            "relative_risk": round(rel_risk, 3),
            "odds_ratio": round(odds_ratio, 3)
        },
        "power_analysis": {
            "observed_power": round(power_est, 3),
            "observed_power_pct": round(power_est * 100.0, 1),
            "required_n_per_variant": req_n_per_variant
        },
        "is_significant": is_significant,
        "winner": winner,
        "status_tag": status_tag,
        "conclusion": conclusion
    }


def run_ab_mean_test(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    control_val: Any,
    treatment_val: Any,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Executes a Welch's t-test for continuous numeric metrics (Revenue, AOV, Time, Visits).
    Handles unequal sample sizes and variances.
    """
    if group_col not in df.columns or target_col not in df.columns:
        return {"error": f"Columns '{group_col}' or '{target_col}' not found in dataset."}

    sub = df[[group_col, target_col]].dropna()
    c_series = pd.to_numeric(sub[sub[group_col].astype(str) == str(control_val)][target_col], errors='coerce').dropna()
    t_series = pd.to_numeric(sub[sub[group_col].astype(str) == str(treatment_val)][target_col], errors='coerce').dropna()

    n_c = len(c_series)
    n_t = len(t_series)

    if n_c < 5 or n_t < 5:
        return {"error": f"Insufficient sample size (Control: {n_c}, Treatment: {n_t}). Minimum 5 numeric observations required per variant."}

    m_c = float(c_series.mean())
    m_t = float(t_series.mean())

    s_c = float(c_series.std(ddof=1)) if n_c > 1 else 0.0
    s_t = float(t_series.std(ddof=1)) if n_t > 1 else 0.0

    se_c = s_c / math.sqrt(n_c) if n_c > 0 else 0.0
    se_t = s_t / math.sqrt(n_t) if n_t > 0 else 0.0

    # Difference & Relative Lift
    diff = m_t - m_c
    rel_lift = (diff / m_c * 100.0) if abs(m_c) > 1e-9 else 0.0

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(t_series, c_series, equal_var=False)
    p_val = max(0.000001, min(1.0, float(p_val)))

    # Welch-Satterthwaite degrees of freedom
    num_df = (s_c**2 / n_c + s_t**2 / n_t) ** 2
    denom_df = ((s_c**2 / n_c)**2 / (n_c - 1)) + ((s_t**2 / n_t)**2 / (n_t - 1)) if (n_c > 1 and n_t > 1) else 1.0
    df_welch = num_df / denom_df if denom_df > 0 else (n_c + n_t - 2)

    # Confidence interval of difference
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_welch))
    se_diff = math.sqrt((s_c**2 / n_c) + (s_t**2 / n_t)) if (n_c > 0 and n_t > 0) else 0.0001
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    # Effect Size: Cohen's d (pooled standard deviation)
    s_pooled = math.sqrt(((n_c - 1) * (s_c ** 2) + (n_t - 1) * (s_t ** 2)) / (n_c + n_t - 2)) if (n_c + n_t > 2) else (s_c + s_t)/2
    cohen_d = abs(diff) / s_pooled if s_pooled > 0 else 0.0
    d_label = "Negligible" if cohen_d < 0.2 else ("Small" if cohen_d < 0.5 else ("Medium" if cohen_d < 0.8 else "Large"))

    # Observed statistical power
    z_equiv = abs(t_stat)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    power_est = float(stats.norm.cdf(z_equiv - z_crit)) if z_equiv > 0 else 0.5
    power_est = max(0.05, min(0.999, power_est))

    # Required sample size per variant for 80% power at alpha=0.05
    z_beta_80 = 0.8416
    if cohen_d > 0.005:
        req_n_per_variant = int(math.ceil(2.0 * ((z_crit + z_beta_80) ** 2) / (cohen_d ** 2)))
    else:
        req_n_per_variant = 50000

    is_significant = bool(p_val <= alpha)
    if is_significant:
        if diff > 0:
            winner = "Treatment (Variant)"
            status_tag = "Significant Winner"
            conclusion = (
                f"Treatment generated a statistically significant mean lift of +{rel_lift:.2f}% "
                f"({m_c:,.2f} vs {m_t:,.2f}, p = {p_val:.4f} < {alpha:.2f}). "
                f"We are {int(confidence_level*100)}% confident that the true average improvement is between "
                f"{ci_low:+,.2f} and {ci_high:+,.2f} units. Recommendation: Roll out Treatment to all users."
            )
        else:
            winner = "Control (Baseline)"
            status_tag = "Significant Underperformer"
            conclusion = (
                f"Treatment showed a statistically significant performance decline of {rel_lift:.2f}% "
                f"({m_t:,.2f} vs {m_c:,.2f}, p = {p_val:.4f} < {alpha:.2f}). "
                f"Recommendation: Keep Control; discontinue Variant."
            )
    else:
        winner = "Inconclusive / No Winner"
        status_tag = "Not Statistically Significant"
        conclusion = (
            f"The observed difference of {diff:+,.2f} units (lift: {rel_lift:+.2f}%) is not statistically significant "
            f"at the {int(confidence_level*100)}% confidence level (p = {p_val:.4f} >= {alpha:.2f}). "
            f"Statistical power is currently {power_est*100:.1f}%. Estimated sample size to detect this effect at 80% power: "
            f"~{req_n_per_variant:,} observations per variant."
        )

    return {
        "test_type": "mean",
        "metric_name": target_col,
        "group_col": group_col,
        "confidence_level": confidence_level,
        "control_name": str(control_val),
        "treatment_name": str(treatment_val),
        "control": {
            "n": n_c,
            "mean": round(m_c, 4),
            "std": round(s_c, 4),
            "se": round(se_c, 4)
        },
        "treatment": {
            "n": n_t,
            "mean": round(m_t, 4),
            "std": round(s_t, 4),
            "se": round(se_t, 4)
        },
        "difference": round(diff, 4),
        "relative_lift_pct": round(rel_lift, 2),
        "t_statistic": round(float(t_stat), 4),
        "df": round(float(df_welch), 2),
        "p_value": round(float(p_val), 6),
        "confidence_interval": {
            "low": round(ci_low, 4),
            "high": round(ci_high, 4)
        },
        "effect_size": {
            "metric": "Cohen's d",
            "value": round(cohen_d, 4),
            "magnitude": d_label,
            "pooled_std": round(s_pooled, 4)
        },
        "power_analysis": {
            "observed_power": round(power_est, 3),
            "observed_power_pct": round(power_est * 100.0, 1),
            "required_n_per_variant": req_n_per_variant
        },
        "is_significant": is_significant,
        "winner": winner,
        "status_tag": status_tag,
        "conclusion": conclusion
    }


def calculate_sample_size(
    baseline_val: float,
    mde_pct: float,
    metric_type: str = "conversion",
    alpha: float = 0.05,
    power: float = 0.80,
    sd: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates required sample size per variant for pre-experiment design.
    """
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    if metric_type == "conversion":
        p1 = baseline_val
        p2 = p1 * (1.0 + mde_pct / 100.0)
        p2 = max(0.0001, min(0.9999, p2))
        diff = abs(p2 - p1)
        if diff < 1e-6:
            return {"error": "Minimum detectable effect cannot be zero."}
        p_avg = (p1 + p2) / 2.0
        n_per_variant = int(math.ceil(2.0 * ((z_alpha + z_beta) ** 2) * p_avg * (1.0 - p_avg) / (diff ** 2)))
        return {
            "metric_type": "conversion",
            "baseline_conversion_rate": round(p1, 4),
            "expected_conversion_rate": round(p2, 4),
            "mde_pct": mde_pct,
            "alpha": alpha,
            "power": power,
            "n_per_variant": n_per_variant,
            "total_sample_size": n_per_variant * 2
        }
    else:
        mean_base = baseline_val
        diff = abs(mean_base * (mde_pct / 100.0))
        std_dev = sd if sd and sd > 0 else (abs(mean_base) * 0.5)
        if diff < 1e-6:
            return {"error": "Minimum detectable effect cannot be zero."}
        cohen_d = diff / std_dev
        n_per_variant = int(math.ceil(2.0 * ((z_alpha + z_beta) ** 2) / (cohen_d ** 2)))
        return {
            "metric_type": "mean",
            "baseline_mean": round(mean_base, 4),
            "expected_mean": round(mean_base + (mean_base * mde_pct / 100.0), 4),
            "standard_deviation": round(std_dev, 4),
            "mde_pct": mde_pct,
            "cohen_d": round(cohen_d, 4),
            "alpha": alpha,
            "power": power,
            "n_per_variant": n_per_variant,
            "total_sample_size": n_per_variant * 2
        }


def generate_sample_ab_dataset(test_type: str = "conversion") -> pd.DataFrame:
    """
    Generates realistic A/B experiment data for demonstration.
    """
    np.random.seed(42)
    if test_type == "conversion":
        n_c = 1200
        n_t = 1200
        c_conv = np.random.binomial(1, 0.102, n_c)
        t_conv = np.random.binomial(1, 0.145, n_t)
        
        c_df = pd.DataFrame({
            "User_ID": [f"USR-CTRL-{i+1:04d}" for i in range(n_c)],
            "Variant": ["Control" for _ in range(n_c)],
            "Device": np.random.choice(["Desktop", "Mobile", "Tablet"], n_c, p=[0.55, 0.38, 0.07]),
            "Converted": c_conv,
            "TimeOnPage_Sec": np.round(np.random.normal(75, 20, n_c).clip(10, 300), 1),
            "Revenue": np.round(c_conv * np.random.exponential(45, n_c) + 0.0, 2)
        })
        t_df = pd.DataFrame({
            "User_ID": [f"USR-TRMT-{i+1:04d}" for i in range(n_t)],
            "Variant": ["Treatment" for _ in range(n_t)],
            "Device": np.random.choice(["Desktop", "Mobile", "Tablet"], n_t, p=[0.55, 0.38, 0.07]),
            "Converted": t_conv,
            "TimeOnPage_Sec": np.round(np.random.normal(82, 22, n_t).clip(10, 300), 1),
            "Revenue": np.round(t_conv * np.random.exponential(48, n_t) + 0.0, 2)
        })
        return pd.concat([c_df, t_df], ignore_index=True)
    else:
        n_c = 800
        n_t = 800
        c_rev = np.random.gamma(shape=3.0, scale=25.0, size=n_c)
        t_rev = np.random.gamma(shape=3.3, scale=26.5, size=n_t)
        
        c_df = pd.DataFrame({
            "Customer_ID": [f"CUST-A-{i+1:04d}" for i in range(n_c)],
            "Pricing_Model": ["Standard" for _ in range(n_c)],
            "Plan_Tier": np.random.choice(["Basic", "Pro", "Enterprise"], n_c, p=[0.6, 0.3, 0.1]),
            "Order_Value": np.round(c_rev, 2),
            "Items_Purchased": np.random.poisson(2.5, n_c) + 1
        })
        t_df = pd.DataFrame({
            "Customer_ID": [f"CUST-B-{i+1:04d}" for i in range(n_t)],
            "Pricing_Model": ["Dynamic_Discount" for _ in range(n_t)],
            "Plan_Tier": np.random.choice(["Basic", "Pro", "Enterprise"], n_t, p=[0.6, 0.3, 0.1]),
            "Order_Value": np.round(t_rev, 2),
            "Items_Purchased": np.random.poisson(2.8, n_t) + 1
        })
        return pd.concat([c_df, t_df], ignore_index=True)
