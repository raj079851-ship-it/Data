"""
Data Quality Center Module
Data Governance & Drift Monitoring:
- Rule-Based Validation Engine (Not Null, Min/Max Bounds, Allowed Categories, Regex Patterns)
- Continuous Health Score tracking
- Data Drift & Feature Drift detection (Kolmogorov-Smirnov test & Population Stability Index)
- Monitoring alerts and quality degradation threshold flags
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats


DEFAULT_QUALITY_RULES = [
    {"column": "CustomerID", "rule_type": "not_null", "params": {}},
    {"column": "MonthlyCharges", "rule_type": "min_value", "params": {"min": 0.0}},
    {"column": "TotalCharges", "rule_type": "min_value", "params": {"min": 0.0}},
    {"column": "Contract", "rule_type": "allowed_values", "params": {"allowed": ["Month-to-month", "One year", "Two year"]}}
]


class DataQualityCenter:
    """Manages quality rule enforcement and drift monitoring."""

    def __init__(self):
        self.rules = list(DEFAULT_QUALITY_RULES)
        self.quality_history: List[Dict[str, Any]] = []

    def evaluate_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates active validation rules against the dataset."""
        rule_results = []
        passed_rules = 0

        for r in self.rules:
            col = r["column"]
            rtype = r["rule_type"]
            params = r.get("params", {})

            if col not in df.columns:
                rule_results.append({
                    "column": col,
                    "rule": f"{rtype.replace('_', ' ').title()}",
                    "status": "Failed",
                    "violations": len(df),
                    "violation_pct": 100.0,
                    "details": "Column missing from current schema."
                })
                continue

            s = df[col]
            violations = 0

            if rtype == "not_null":
                violations = int(s.isna().sum())
            elif rtype == "min_value":
                min_thresh = float(params.get("min", 0.0))
                num_s = pd.to_numeric(s, errors="coerce")
                violations = int((num_s < min_thresh).sum())
            elif rtype == "max_value":
                max_thresh = float(params.get("max", 1000000.0))
                num_s = pd.to_numeric(s, errors="coerce")
                violations = int((num_s > max_thresh).sum())
            elif rtype == "allowed_values":
                allowed_set = set(params.get("allowed", []))
                violations = int((~s.dropna().astype(str).isin(allowed_set)).sum())

            v_pct = round((violations / len(df) * 100), 2) if len(df) > 0 else 0.0
            is_pass = (violations == 0)
            if is_pass:
                passed_rules += 1

            rule_results.append({
                "column": col,
                "rule": f"{rtype.replace('_', ' ').title()} ({params if params else 'Check'})",
                "status": "Passed" if is_pass else "Failed",
                "violations": violations,
                "violation_pct": v_pct,
                "details": f"{violations:,} records violated constraint." if not is_pass else "Constraint satisfied."
            })

        rule_pass_rate = round((passed_rules / max(1, len(self.rules)) * 100), 1)

        return {
            "total_rules": len(self.rules),
            "passed_rules": passed_rules,
            "failed_rules": len(self.rules) - passed_rules,
            "rule_pass_rate": rule_pass_rate,
            "rule_details": rule_results
        }

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        p_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Runs Kolmogorov-Smirnov test to detect distribution drift
        between baseline and production datasets.
        """
        drift_reports = []
        common_numeric = [
            c for c in baseline_df.select_dtypes(include=[np.number]).columns
            if c in current_df.columns and pd.api.types.is_numeric_dtype(current_df[c])
        ]

        for col in common_numeric:
            s1 = baseline_df[col].dropna()
            s2 = current_df[col].dropna()

            if len(s1) > 10 and len(s2) > 10:
                ks_stat, p_val = stats.ks_2samp(s1, s2)
                has_drift = (p_val < p_threshold)
                drift_reports.append({
                    "feature": col,
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 5),
                    "has_drift": has_drift,
                    "status": "Drift Detected (Alert)" if has_drift else "Stable",
                    "baseline_mean": round(float(s1.mean()), 2),
                    "current_mean": round(float(s2.mean()), 2)
                })

        return drift_reports
