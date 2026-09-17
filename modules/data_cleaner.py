"""
Data Cleaner Module
Provides comprehensive data quality auditing, missing value imputation (KNN, AI-recommended,
Mean/Median/Mode, Interpolate, Ffill/Bfill), advanced outlier detection (IQR, Z-score, Modified Z-score,
Isolation Forest, Local Outlier Factor), text & categorical standardization, and Undo/Redo cleaning pipeline.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a deep data quality audit on the DataFrame.
    Returns structured metrics, issues found, and quality health score (0-100).
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    if total_rows == 0 or total_cols == 0:
        return {"health_score": 0, "issues": ["Dataset is empty"]}

    issues = []
    
    # 1. Missing values
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()
    missing_summary = []
    for col, count in cols_with_missing.items():
        pct = round(count / total_rows * 100, 2)
        missing_summary.append({
            "column": col,
            "missing_count": int(count),
            "missing_pct": pct,
            "dtype": str(df[col].dtype)
        })
        if pct > 40:
            issues.append(f"High missing rate in column '{col}' ({pct}% missing).")

    # 2. Duplicate rows
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / total_rows * 100, 2)
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicated rows ({dup_pct}% of dataset).")

    # 3. Constant columns (cardinality 1)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        issues.append(f"Found {len(constant_cols)} constant/single-value column(s): {', '.join(constant_cols)}.")

    # 4. Outliers detection across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    for col in numeric_cols:
        valid_series = df[col].dropna()
        if len(valid_series) < 5:
            continue
        q25 = valid_series.quantile(0.25)
        q75 = valid_series.quantile(0.75)
        iqr = q75 - q25
        lower_iqr = q25 - 1.5 * iqr
        upper_iqr = q75 + 1.5 * iqr
        iqr_outliers = int(((valid_series < lower_iqr) | (valid_series > upper_iqr)).sum())
        
        std = valid_series.std()
        mean = valid_series.mean()
        z_outliers = 0
        if std > 0:
            z_scores = np.abs((valid_series - mean) / std)
            z_outliers = int((z_scores > 3.0).sum())

        if iqr_outliers > 0 or z_outliers > 0:
            outlier_summary[col] = {
                "iqr_outliers": iqr_outliers,
                "iqr_pct": round(iqr_outliers / len(valid_series) * 100, 2),
                "lower_bound": round(float(lower_iqr), 3),
                "upper_bound": round(float(upper_iqr), 3),
                "z_score_outliers": z_outliers
            }
            if round(iqr_outliers / len(valid_series) * 100, 2) > 10:
                issues.append(f"High outlier volume in numeric column '{col}' ({iqr_outliers} IQR outliers).")

    # 5. String whitespaces
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    whitespace_cols = []
    for col in string_cols:
        s = df[col].dropna().astype(str)
        has_padded_spaces = s.str.startswith(" ") | s.str.endswith(" ")
        if has_padded_spaces.any():
            whitespace_cols.append(col)
    if whitespace_cols:
        issues.append(f"Found untrimmed whitespace in columns: {', '.join(whitespace_cols)}.")

    # 6. Overall Health Score
    score = 100.0
    total_missing = missing_counts.sum()
    total_cells = total_rows * total_cols
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0
    score -= min(40, missing_ratio * 150)
    score -= min(25, dup_pct * 2.5)
    score -= min(15, len(constant_cols) * 5)
    
    total_outliers = sum(v["iqr_outliers"] for v in outlier_summary.values())
    outlier_ratio = total_outliers / (total_rows * max(1, len(numeric_cols))) if len(numeric_cols) > 0 else 0
    score -= min(15, outlier_ratio * 100)
    score = max(5, int(round(score)))

    return {
        "health_score": score,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_summary": missing_summary,
        "total_missing_cells": int(total_missing),
        "missing_pct": round(missing_ratio * 100, 2),
        "duplicated_rows": dup_count,
        "duplicated_pct": dup_pct,
        "constant_columns": constant_cols,
        "outlier_summary": outlier_summary,
        "whitespace_columns": whitespace_cols,
        "issues": issues if issues else ["No major data hygiene issues detected."]
    }


def impute_missing_values(
    df: pd.DataFrame,
    strategy_dict: Optional[Dict[str, str]] = None,
    constant_values: Optional[Dict[str, Any]] = None,
    knn_neighbors: int = 5
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Advanced missing value imputation supporting:
    - mean, median, mode
    - forward_fill, backward_fill, constant
    - interpolate (linear)
    - knn (K-Nearest Neighbors for numeric features)
    - ai_recommended (heuristically chooses best method based on skewness/distribution)
    - drop_row, drop_col
    Returns (cleaned_df, changelog, stats_before_after).
    """
    df_clean = df.copy()
    changelog = []
    before_stats = df.isna().sum().to_dict()

    if strategy_dict is None:
        strategy_dict = {}
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                strategy_dict[col] = "ai_recommended"

    cols_for_knn = []

    for col, strategy in strategy_dict.items():
        if isinstance(strategy, dict):
            strategy = strategy.get("strategy", "ai_recommended")
        if col not in df_clean.columns:
            continue
        missing_count = int(df_clean[col].isna().sum())
        if missing_count == 0:
            continue

        # AI-recommended logic
        if strategy == "ai_recommended":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                skew = abs(df_clean[col].skew()) if len(df_clean[col].dropna()) > 3 else 0
                if skew > 1.0:
                    strategy = "median"  # Skewed -> median is robust
                elif df_clean[col].nunique() < 10:
                    strategy = "mode"
                else:
                    strategy = "median"
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                strategy = "forward_fill"
            else:
                strategy = "mode"

        if strategy == "drop_col":
            df_clean = df_clean.drop(columns=[col])
            changelog.append(f"Dropped column '{col}' with {missing_count} missing values.")
        elif strategy == "drop_row":
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(subset=[col])
            rows_dropped = rows_before - len(df_clean)
            changelog.append(f"Dropped {rows_dropped} rows where '{col}' was missing.")
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
            mean_val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(mean_val)
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Mean ({round(mean_val, 2)}).")
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
            med_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med_val)
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Median ({round(med_val, 2)}).")
        elif strategy == "mode":
            mode_series = df_clean[col].mode(dropna=True)
            mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Mode ('{mode_val}').")
        elif strategy == "forward_fill":
            df_clean[col] = df_clean[col].ffill()
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Forward Fill.")
        elif strategy == "backward_fill":
            df_clean[col] = df_clean[col].bfill()
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Backward Fill.")
        elif strategy == "interpolate" and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].interpolate(method="linear")
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with Linear Interpolation.")
        elif strategy == "constant":
            c_val = "Missing"
            if constant_values and col in constant_values:
                c_val = constant_values[col]
            df_clean[col] = df_clean[col].fillna(c_val)
            changelog.append(f"Imputed {missing_count} missing values in '{col}' with constant '{c_val}'.")
        elif strategy == "knn" and pd.api.types.is_numeric_dtype(df_clean[col]):
            cols_for_knn.append(col)

    # Perform KNN Imputation across numeric features if requested
    if cols_for_knn:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            try:
                k = min(knn_neighbors, max(1, len(df_clean) - 1))
                imputer = KNNImputer(n_neighbors=k)
                imputed_arr = imputer.fit_transform(df_clean[num_cols])
                df_clean[num_cols] = imputed_arr
                changelog.append(f"Performed KNN Imputation (k={k}) on columns: {', '.join(cols_for_knn)}.")
            except Exception as e:
                # Fallback to median
                for c in cols_for_knn:
                    df_clean[c] = df_clean[c].fillna(df_clean[c].median())
                changelog.append(f"KNN imputation fell back to median: {str(e)}")

    return df_clean, changelog


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",       # 'iqr', 'z_score', 'modified_z_score', 'isolation_forest', 'lof'
    action: str = "cap",       # 'cap', 'remove', 'set_nan', 'keep'
    factor: float = 1.5,
    z_thresh: float = 3.0,
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Advanced Outlier Detection & Treatment:
    - IQR (Interquartile Range)
    - Z-Score
    - Modified Z-Score (Median Absolute Deviation)
    - Isolation Forest
    - Local Outlier Factor (LOF)
    Actions: 'cap' (Winsorize), 'remove' (drop rows), 'set_nan', 'keep'
    """
    df_out = df.copy()
    changelog = []
    
    if columns is None:
        columns = df_out.select_dtypes(include=[np.number]).columns.tolist()

    if action == "keep":
        return df_out, ["Kept outliers without modification."]

    # Machine Learning Outlier Methods (Multi-column)
    if method in ["isolation_forest", "lof"] and len(columns) > 0:
        valid_cols = [c for c in columns if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
        if valid_cols:
            sub = df_out[valid_cols].dropna()
            if len(sub) > 10:
                try:
                    if method == "isolation_forest":
                        clf = IsolationForest(contamination=contamination, random_state=42)
                        preds = clf.fit_predict(sub)
                    else:
                        clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                        preds = clf.fit_predict(sub)
                    
                    outlier_indices = sub.index[preds == -1]
                    count = len(outlier_indices)
                    if count > 0:
                        if action == "remove":
                            df_out = df_out.drop(index=outlier_indices).reset_index(drop=True)
                            changelog.append(f"Removed {count} anomalous rows detected by {method.replace('_', ' ').title()}.")
                        elif action == "set_nan":
                            for col in valid_cols:
                                df_out.loc[outlier_indices, col] = np.nan
                            changelog.append(f"Flagged {count} rows as NaN via {method.replace('_', ' ').title()}.")
                        elif action == "cap":
                            # Winsorize valid cols to 1st and 99th percentiles
                            for col in valid_cols:
                                p1, p99 = df_out[col].quantile(0.01), df_out[col].quantile(0.99)
                                df_out[col] = df_out[col].clip(lower=p1, upper=p99)
                            changelog.append(f"Winsorized features across {count} anomalies using {method.replace('_', ' ').title()}.")
                    return df_out, changelog
                except Exception as e:
                    changelog.append(f"ML Outlier detection error: {str(e)}. Falling back to IQR.")
                    method = "iqr"

    # Univariate methods (IQR, Z-Score, Modified Z-Score)
    for col in columns:
        if col not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        valid_data = df_out[col].dropna()
        if len(valid_data) < 5:
            continue

        if method == "iqr":
            q25 = valid_data.quantile(0.25)
            q75 = valid_data.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - factor * iqr
            upper_bound = q75 + factor * iqr
        elif method == "modified_z_score":
            median = valid_data.median()
            mad = (valid_data - median).abs().median()
            if mad == 0:
                continue
            lower_bound = median - (3.5 * mad / 0.6745)
            upper_bound = median + (3.5 * mad / 0.6745)
        else:  # z_score
            mean = valid_data.mean()
            std = valid_data.std()
            if std == 0:
                continue
            lower_bound = mean - z_thresh * std
            upper_bound = mean + z_thresh * std

        outlier_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        count = int(outlier_mask.sum())
        if count == 0:
            continue

        if action == "cap":
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            changelog.append(
                f"Capped {count} outliers in '{col}' via {method.upper()} to [{round(lower_bound, 2)}, {round(upper_bound, 2)}]."
            )
        elif action == "remove":
            df_out = df_out[~outlier_mask].reset_index(drop=True)
            changelog.append(f"Removed {count} rows containing outliers in '{col}' ({method.upper()}).")
        elif action == "set_nan":
            df_out.loc[outlier_mask, col] = np.nan
            changelog.append(f"Set {count} outliers in '{col}' to NaN.")

    return df_out, changelog


def clean_text_and_duplicates(
    df: pd.DataFrame,
    remove_dups: bool = True,
    keep_dup_mode: str = "first", # 'first', 'last', 'drop_all'
    trim_strings: bool = True,
    text_case: str = "keep",      # 'keep', 'lower', 'upper', 'title'
    remove_special_chars: bool = False,
    replace_dict: Optional[Dict[str, str]] = None,
    regex_replace: Optional[Tuple[str, str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Deduplicates records and standardizes text columns."""
    df_clean = df.copy()
    changelog = []

    if remove_dups:
        dups = int(df_clean.duplicated().sum())
        if dups > 0:
            keep_arg = False if keep_dup_mode == "drop_all" else keep_dup_mode
            df_clean = df_clean.drop_duplicates(keep=keep_arg).reset_index(drop=True)
            changelog.append(f"Removed {dups} duplicate rows (keep='{keep_dup_mode}').")

    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        s = df_clean[col].copy()
        if trim_strings:
            s = s.apply(lambda x: x.strip() if isinstance(x, str) else x)
        if text_case == "lower":
            s = s.apply(lambda x: x.lower() if isinstance(x, str) else x)
        elif text_case == "upper":
            s = s.apply(lambda x: x.upper() if isinstance(x, str) else x)
        elif text_case == "title":
            s = s.apply(lambda x: x.title() if isinstance(x, str) else x)
        
        if remove_special_chars:
            s = s.apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x) if isinstance(x, str) else x)
            
        if replace_dict:
            s = s.replace(replace_dict)
            
        if regex_replace:
            pattern, repl = regex_replace
            s = s.apply(lambda x: re.sub(pattern, repl, str(x)) if isinstance(x, str) and pd.notna(x) else x)
            
        df_clean[col] = s

    if trim_strings and len(str_cols) > 0:
        changelog.append(f"Trimmed excess whitespace from {len(str_cols)} text columns.")
    if text_case != "keep":
        changelog.append(f"Normalized case to '{text_case}'.")
    if remove_special_chars:
        changelog.append("Removed special non-alphanumeric characters.")
    if regex_replace:
        changelog.append(f"Applied regex replacement '{regex_replace[0]}' -> '{regex_replace[1]}'.")

    return df_clean, changelog


def convert_column_types(
    df: pd.DataFrame,
    type_map: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
    """type_map: {'col': 'datetime' | 'numeric' | 'categorical' | 'string' | 'bool'}"""
    df_converted = df.copy()
    changelog = []

    for col, target_type in type_map.items():
        if col not in df_converted.columns:
            continue
        try:
            if target_type == "datetime":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
                changelog.append(f"Converted '{col}' to datetime.")
            elif target_type == "numeric":
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
                changelog.append(f"Converted '{col}' to numeric.")
            elif target_type == "categorical":
                df_converted[col] = df_converted[col].astype("category")
                changelog.append(f"Converted '{col}' to categorical.")
            elif target_type == "string":
                df_converted[col] = df_converted[col].astype(str)
                changelog.append(f"Converted '{col}' to string.")
            elif target_type == "bool":
                df_converted[col] = df_converted[col].astype(bool)
                changelog.append(f"Converted '{col}' to boolean.")
        except Exception as e:
            changelog.append(f"Failed to cast '{col}' to {target_type}: {str(e)}")

    return df_converted, changelog


def one_click_ai_auto_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Automated high-impact cleaning pipeline:
    1. Removes duplicate rows
    2. Drops constant / single-value columns
    3. Normalizes whitespace across strings
    4. Imputes missing numeric values with Median & categoricals with Mode
    5. Winsorizes (caps) extreme outliers via IQR (3.0x threshold)
    """
    df_clean = df.copy()
    changelog = []

    # 1. Duplicates
    df_clean, dups_log = clean_text_and_duplicates(df_clean, remove_dups=True, trim_strings=True)
    changelog.extend(dups_log)

    # 2. Constant columns
    constant_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df_clean = df_clean.drop(columns=constant_cols)
        changelog.append(f"Dropped {len(constant_cols)} uninformative constant column(s): {', '.join(constant_cols)}.")

    # 3. Imputation
    df_clean, imp_log = impute_missing_values(df_clean)
    changelog.extend(imp_log)

    # 4. Outlier capping
    df_clean, outlier_log = handle_outliers(df_clean, method="iqr", action="cap", factor=2.5)
    changelog.extend(outlier_log)

    return df_clean, changelog


class CleaningPipelineManager:
    """Maintains an undo/redo stack of cleaning operations."""

    def __init__(self, initial_df: pd.DataFrame):
        self.history: List[Tuple[str, pd.DataFrame]] = [("Initial State", initial_df.copy())]
        self.current_idx: int = 0

    def apply_step(self, name: str, new_df: pd.DataFrame):
        # Truncate any redo branch
        self.history = self.history[:self.current_idx + 1]
        self.history.append((name, new_df.copy()))
        self.current_idx += 1

    def can_undo(self) -> bool:
        return self.current_idx > 0

    def can_redo(self) -> bool:
        return self.current_idx < len(self.history) - 1

    def undo(self) -> Optional[Tuple[str, pd.DataFrame]]:
        if self.can_undo():
            self.current_idx -= 1
            return self.history[self.current_idx]
        return None

    def redo(self) -> Optional[Tuple[str, pd.DataFrame]]:
        if self.can_redo():
            self.current_idx += 1
            return self.history[self.current_idx]
        return None

    def get_current(self) -> pd.DataFrame:
        return self.history[self.current_idx][1].copy()

    def get_pipeline_steps(self) -> List[str]:
        return [step[0] for step in self.history]
