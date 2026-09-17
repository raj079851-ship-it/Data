"""
Feature Engineering Module
Provides automated column categorization, AI feature suggestions, scaling,
encoding, datetime decomposition, interaction feature synthesis, polynomial terms,
rolling statistics, target encoding, and text feature extraction.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures


def categorize_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classifies columns into:
    - numerical: float/int continuous and discrete
    - categorical_low: <= 10 unique values
    - categorical_high: > 10 unique values
    - datetime: datetime dtype or parseable date string
    - text_or_id: high cardinality identifiers or free text
    """
    numerical = []
    categorical_low = []
    categorical_high = []
    datetime_cols = []
    text_or_id = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and (
                "date" in col.lower() or "time" in col.lower() or sample.str.match(r"^\d{4}-\d{2}-\d{2}").all()
            ):
                try:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
                    continue
                except Exception:
                    pass

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == len(df) and len(df) > 50 and df[col].dtype in [np.int64, np.int32]:
                text_or_id.append(col)
            else:
                numerical.append(col)
        else:
            unique_count = df[col].nunique(dropna=True)
            ratio_unique = unique_count / len(df) if len(df) > 0 else 0
            
            if ratio_unique > 0.85 and unique_count > 50:
                text_or_id.append(col)
            elif unique_count <= 10:
                categorical_low.append(col)
            else:
                categorical_high.append(col)

    return {
        "numerical": numerical,
        "categorical_low": categorical_low,
        "categorical_high": categorical_high,
        "datetime": datetime_cols,
        "text_or_id": text_or_id
    }


def get_feature_recommendations(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generates intelligent feature engineering suggestions with rationale and impacts."""
    recs = []
    cat_info = categorize_features(df)

    # 1. Skewed numerical features
    for col in cat_info["numerical"]:
        series = df[col].dropna()
        if len(series) > 10:
            skew = series.skew()
            if abs(skew) > 1.2 and (series >= 0).all():
                recs.append({
                    "column": col,
                    "type": "Log Transform",
                    "suggestion": f"Apply Log(x + 1) transform on '{col}'",
                    "reason": f"High skewness ({round(skew, 2)}) detected. Stabilizes variance for linear and neural models."
                })

    # 2. Datetime features
    for col in cat_info["datetime"]:
        recs.append({
            "column": col,
            "type": "Date Decomposition",
            "suggestion": f"Extract Year, Month, Day, DayOfWeek, Is_Weekend, and Quarter from '{col}'",
            "reason": "Decomposing timestamps uncovers seasonal trends and cyclical behavior."
        })

    # 3. Categorical low cardinality
    for col in cat_info["categorical_low"]:
        recs.append({
            "column": col,
            "type": "One-Hot Encoding",
            "suggestion": f"One-Hot Encode '{col}' ({df[col].nunique()} categories)",
            "reason": "Clean low cardinality feature ideal for dummy variable encoding."
        })

    # 4. Categorical high cardinality
    for col in cat_info["categorical_high"]:
        recs.append({
            "column": col,
            "type": "Frequency / Target Encoding",
            "suggestion": f"Use Frequency or Target Encoding for '{col}' ({df[col].nunique()} categories)",
            "reason": "Avoids dimensionality explosion while retaining categorical prevalence."
        })

    # 5. Domain interactions
    num_cols = cat_info["numerical"]
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                if ("charge" in c1.lower() and "month" in c2.lower()) or ("tenure" in c2.lower() and "charge" in c1.lower()):
                    recs.append({
                        "column": f"{c1} & {c2}",
                        "type": "Feature Interaction / Ratio",
                        "suggestion": f"Create Ratio feature: {c1} / ({c2} + 1)",
                        "reason": "Captures normalized expenditure per unit of tenure."
                    })
                elif ("sqft" in c1.lower() and "bedroom" in c2.lower()) or ("price" in c1.lower() and "sqft" in c2.lower()):
                    recs.append({
                        "column": f"{c1} & {c2}",
                        "type": "Feature Interaction / Ratio",
                        "suggestion": f"Create Ratio feature: {c1} / {c2}",
                        "reason": "Represents per-unit metric (e.g. price per sqft), which is highly predictive."
                    })
                elif ("revenue" in c1.lower() and "cost" in c2.lower()):
                    recs.append({
                        "column": f"{c1} & {c2}",
                        "type": "Profit / Margin Difference",
                        "suggestion": f"Create Difference feature: {c1} - {c2}",
                        "reason": "Calculates gross operating profit directly."
                    })

    return recs


def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    replace: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Scales numerical features using Standard, MinMax, or Robust scaler."""
    df_out = df.copy()
    changelog = []

    if not columns:
        return df_out, changelog

    valid_cols = []
    for c in columns:
        if c in df_out.columns:
            if not pd.api.types.is_numeric_dtype(df_out[c]):
                try:
                    df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
                except Exception:
                    continue
            valid_cols.append(c)

    if not valid_cols:
        return df_out, changelog

    if method == "standard":
        scaler = StandardScaler()
        suffix = "_std"
    elif method == "minmax":
        scaler = MinMaxScaler()
        suffix = "_minmax"
    elif method == "robust":
        scaler = RobustScaler()
        suffix = "_robust"
    else:
        scaler = StandardScaler()
        suffix = "_scaled"

    for col in valid_cols:
        col_series = pd.to_numeric(df_out[col], errors="coerce")
        median_val = col_series.median()
        fill_val = 0.0 if pd.isna(median_val) else float(median_val)
        clean_col = col_series.fillna(fill_val).to_numpy().reshape(-1, 1)

        try:
            scaled_vals = scaler.fit_transform(clean_col).flatten()
            target_col = col if replace else f"{col}{suffix}"
            if not replace and target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])
            df_out[target_col] = np.round(scaled_vals, 4)
            changelog.append(f"Scaled '{col}' via {method.title()} Scaler -> '{target_col}'.")
        except Exception as e:
            changelog.append(f"Could not scale '{col}': {e}")

    return df_out, changelog


def transform_numerical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "log1p",
    bins: int = 5,
    bin_strategy: str = "quantile",
    replace: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Non-linear transformations: log1p, sqrt, or binning."""
    df_out = df.copy()
    changelog = []

    for col in columns:
        if col not in df_out.columns:
            continue

        col_series = pd.to_numeric(df_out[col], errors="coerce")

        if method == "log1p":
            target_col = col if replace else f"{col}_log1p"
            min_val = col_series.min(skipna=True)
            shift = abs(min_val) if (min_val is not None and not pd.isna(min_val) and min_val < 0) else 0.0
            if not replace and target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])
            df_out[target_col] = np.round(np.log1p(np.maximum(col_series + shift, 0)), 4)
            changelog.append(f"Applied Log1p transformation to '{col}' (shift={round(shift, 2)}) -> '{target_col}'.")
        elif method == "sqrt":
            target_col = col if replace else f"{col}_sqrt"
            if not replace and target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])
            df_out[target_col] = np.round(np.sqrt(np.maximum(col_series, 0)), 4)
            changelog.append(f"Applied Square Root transformation to '{col}' -> '{target_col}'.")
        elif method == "binning":
            target_col = col if replace else f"{col}_binned"
            if not replace and target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])
            try:
                clean_s = col_series.dropna()
                actual_bins = max(2, min(bins, len(clean_s.unique())))
                labels = [f"Bin_{i+1}" for i in range(actual_bins)]
                if bin_strategy == "quantile":
                    binned = pd.qcut(col_series, q=actual_bins, labels=labels, duplicates="drop")
                else:
                    binned = pd.cut(col_series, bins=actual_bins, labels=labels)
                df_out[target_col] = binned.astype(str).replace("nan", np.nan)
                changelog.append(f"Binned '{col}' into {actual_bins} intervals ({bin_strategy}) -> '{target_col}'.")
            except Exception as e:
                changelog.append(f"Could not bin '{col}': {e}")

    return df_out, changelog


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "one_hot",
    drop_first: bool = False,
    max_categories: int = 10,
    drop_original: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Encodes categorical features via One-Hot, Label, or Frequency encoding."""
    df_out = df.copy()
    changelog = []

    for col in columns:
        if col not in df_out.columns:
            continue

        if method == "one_hot":
            val_counts = df_out[col].astype(str).value_counts(dropna=False)
            if len(val_counts) > max_categories:
                top_cats = val_counts.nlargest(max_categories).index
                col_series = df_out[col].astype(str).apply(lambda x: x if x in top_cats else "Other")
            else:
                col_series = df_out[col].astype(str)

            safe_prefix = "".join(c if c.isalnum() else "_" for c in str(col))
            dummies = pd.get_dummies(col_series, prefix=safe_prefix, drop_first=drop_first, dtype=int)
            clean_dummy_cols = [
                "".join(c if (c.isalnum() or c == "_") else "_" for c in str(dc))
                for dc in dummies.columns
            ]
            dummies.columns = clean_dummy_cols

            overlap_cols = [c for c in dummies.columns if c in df_out.columns]
            if overlap_cols:
                df_out = df_out.drop(columns=overlap_cols)

            df_out = pd.concat([df_out, dummies], axis=1)
            changelog.append(f"One-Hot Encoded '{col}' into {len(dummies.columns)} binary columns.")

            if drop_original:
                df_out = df_out.drop(columns=[col])
                changelog.append(f"Dropped original column '{col}' after one-hot encoding.")

        elif method == "label":
            le = LabelEncoder()
            valid_mask = df_out[col].notna()
            target_col = f"{col}_encoded"
            if target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])

            encoded_series = pd.Series(index=df_out.index, dtype=float)
            if valid_mask.any():
                encoded_series.loc[valid_mask] = le.fit_transform(df_out.loc[valid_mask, col].astype(str))
            df_out[target_col] = encoded_series
            changelog.append(f"Label Encoded '{col}' -> '{target_col}' ({len(le.classes_)} unique labels).")

            if drop_original:
                df_out = df_out.drop(columns=[col])

        elif method == "frequency":
            freq = df_out[col].value_counts(normalize=True).to_dict()
            target_col = f"{col}_freq"
            if target_col in df_out.columns:
                df_out = df_out.drop(columns=[target_col])
            df_out[target_col] = np.round(df_out[col].map(freq), 4)
            changelog.append(f"Frequency Encoded '{col}' -> '{target_col}'.")

            if drop_original:
                df_out = df_out.drop(columns=[col])

    return df_out, changelog


def target_encode(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str,
    smoothing: float = 10.0
) -> Tuple[pd.DataFrame, str]:
    """Applies smoothed target encoding for high-cardinality categorical features."""
    out = df.copy()
    if categorical_col not in out.columns or target_col not in out.columns:
        return out, "Columns not found."

    global_mean = out[target_col].mean()
    stats = out.groupby(categorical_col)[target_col].agg(["count", "mean"])
    # Smoothed target mean = (count * mean + smoothing * global_mean) / (count + smoothing)
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    new_name = f"{categorical_col}_target_enc"
    out[new_name] = out[categorical_col].map(smoothed).fillna(global_mean)
    return out, f"Target Encoded '{categorical_col}' relative to '{target_col}' -> '{new_name}'."


def extract_datetime_features(
    df: pd.DataFrame,
    columns: List[str],
    drop_original: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Extracts datetime components: Year, Month, Day, DayOfWeek, Is_Weekend, Quarter, Hour, Season."""
    df_out = df.copy()
    changelog = []

    for col in columns:
        if col not in df_out.columns:
            continue

        series = pd.to_datetime(df_out[col], errors="coerce")
        if series.isna().all():
            series = pd.to_datetime(df_out[col], errors="coerce", dayfirst=True)
            if series.isna().all():
                changelog.append(f"Could not parse '{col}' as datetime.")
                continue

        prefix = "".join(c if c.isalnum() else "_" for c in str(col))
        df_out[f"{prefix}_year"] = series.dt.year
        df_out[f"{prefix}_month"] = series.dt.month
        df_out[f"{prefix}_day"] = series.dt.day
        df_out[f"{prefix}_dayofweek"] = series.dt.dayofweek
        df_out[f"{prefix}_is_weekend"] = series.dt.dayofweek.isin([5, 6]).astype(int)
        df_out[f"{prefix}_quarter"] = series.dt.quarter
        
        # Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
        month = series.dt.month
        df_out[f"{prefix}_season"] = np.where(month.isin([12, 1, 2]), "Winter",
                                    np.where(month.isin([3, 4, 5]), "Spring",
                                    np.where(month.isin([6, 7, 8]), "Summer", "Fall")))

        if series.dt.hour.nunique(dropna=True) > 1:
            df_out[f"{prefix}_hour"] = series.dt.hour

        changelog.append(f"Extracted datetime components from '{col}'.")

        if drop_original:
            df_out = df_out.drop(columns=[col])

    return df_out, changelog


def create_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    operation: str = "ratio",
    new_col_name: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Synthesizes interaction features between two numerical columns."""
    df_out = df.copy()
    changelog = []

    if col1 not in df_out.columns or col2 not in df_out.columns:
        return df_out, changelog

    s1 = pd.to_numeric(df_out[col1], errors="coerce")
    s2 = pd.to_numeric(df_out[col2], errors="coerce")

    clean_name = None
    if new_col_name and str(new_col_name).strip():
        clean_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(new_col_name).strip())

    if operation == "ratio":
        name = clean_name or f"{col1}_per_{col2}"
        s2_safe = s2.replace(0, np.nan)
        df_out[name] = np.round(s1 / s2_safe, 4)
        changelog.append(f"Created Ratio feature '{name}' = {col1} / {col2}.")
    elif operation == "multiply":
        name = clean_name or f"{col1}_x_{col2}"
        df_out[name] = np.round(s1 * s2, 4)
        changelog.append(f"Created Product feature '{name}' = {col1} * {col2}.")
    elif operation == "add":
        name = clean_name or f"{col1}_plus_{col2}"
        df_out[name] = np.round(s1 + s2, 4)
        changelog.append(f"Created Sum feature '{name}' = {col1} + {col2}.")
    elif operation == "subtract":
        name = clean_name or f"{col1}_minus_{col2}"
        df_out[name] = np.round(s1 - s2, 4)
        changelog.append(f"Created Difference feature '{name}' = {col1} - {col2}.")

    return df_out, changelog


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2
) -> Tuple[pd.DataFrame, List[str]]:
    """Generates polynomial terms and interaction combinations."""
    out = df.copy()
    valid_cols = [c for c in columns if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    if not valid_cols:
        return out, []

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    sub = out[valid_cols].fillna(out[valid_cols].mean())
    poly_arr = poly.fit_transform(sub)
    poly_names = poly.get_feature_names_out(valid_cols)

    changelog = []
    for i, name in enumerate(poly_names):
        if name not in out.columns:
            clean_name = name.replace(" ", "_").replace("^", "_pow_")
            out[clean_name] = np.round(poly_arr[:, i], 4)
            changelog.append(f"Generated polynomial feature '{clean_name}'.")

    return out, changelog


def calculate_rolling_stats(
    df: pd.DataFrame,
    column: str,
    window: int = 7,
    operations: List[str] = ["mean", "std"]
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculates rolling window statistics and percent change."""
    out = df.copy()
    changelog = []
    if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
        return out, changelog

    s = out[column]
    if "mean" in operations:
        out[f"{column}_roll_mean_{window}"] = np.round(s.rolling(window=window, min_periods=1).mean(), 4)
        changelog.append(f"Computed rolling mean (window={window}) on '{column}'.")
    if "std" in operations:
        out[f"{column}_roll_std_{window}"] = np.round(s.rolling(window=window, min_periods=1).std().fillna(0), 4)
        changelog.append(f"Computed rolling std (window={window}) on '{column}'.")
    if "pct_change" in operations:
        out[f"{column}_pct_change"] = np.round(s.pct_change().fillna(0) * 100, 2)
        changelog.append(f"Computed percent change on '{column}'.")

    return out, changelog


def extract_text_features(
    df: pd.DataFrame,
    column: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Extracts text metadata: character count, word count, uppercase count, digits count, sentiment heuristic."""
    out = df.copy()
    changelog = []
    if column not in out.columns:
        return out, changelog

    s = out[column].astype(str)
    out[f"{column}_char_count"] = s.apply(len)
    out[f"{column}_word_count"] = s.apply(lambda x: len(x.split()))
    out[f"{column}_digits_count"] = s.apply(lambda x: sum(c.isdigit() for c in x))
    
    # Sentiment score heuristic (-1 to +1) based on positive/negative lexical tokens
    pos_words = {"good", "great", "excellent", "amazing", "love", "satisfied", "happy", "fast", "helpful", "pro", "best", "super"}
    neg_words = {"bad", "poor", "terrible", "slow", "broken", "worst", "unhappy", "delay", "error", "fail", "hate", "issue"}
    
    def score_sentiment(text: str) -> float:
        words = set(re.findall(r"\w+", text.lower()))
        pos = len(words & pos_words)
        neg = len(words & neg_words)
        total = pos + neg
        return round((pos - neg) / total, 2) if total > 0 else 0.0

    out[f"{column}_sentiment_score"] = s.apply(score_sentiment)
    changelog.append(f"Extracted char count, word count, digit count, and lexical sentiment score from text column '{column}'.")

    return out, changelog


def one_click_ai_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Automated 1-Click AI Feature Engineering Pipeline."""
    df_out = df.copy()
    changelog = []
    cat_info = categorize_features(df_out)

    if cat_info["datetime"]:
        df_out, dt_log = extract_datetime_features(df_out, cat_info["datetime"][:2], drop_original=False)
        changelog.extend(dt_log)

    recs = get_feature_recommendations(df_out)
    skew_recs = [r for r in recs if r["type"] == "Log Transform"]
    if skew_recs:
        skew_cols = [r["column"] for r in skew_recs[:3] if r["column"] in df_out.columns]
        if skew_cols:
            df_out, tr_log = transform_numerical(df_out, skew_cols, method="log1p", replace=False)
            changelog.extend(tr_log)

    low_cats = [c for c in cat_info["categorical_low"] if c in df_out.columns and df_out[c].nunique() <= 6][:3]
    if low_cats:
        df_out, enc_log = encode_categorical(df_out, low_cats, method="one_hot", max_categories=6, drop_original=False)
        changelog.extend(enc_log)

    curr_nums = df_out.select_dtypes(include=[np.number]).columns.tolist()
    scale_candidates = [c for c in cat_info["numerical"] if c in curr_nums and df_out[c].std() > 1][:3]
    if scale_candidates:
        df_out, sc_log = scale_features(df_out, scale_candidates, method="standard", replace=False)
        changelog.extend(sc_log)

    int_recs = [r for r in recs if "Interaction" in r["type"]]
    if int_recs:
        parts = int_recs[0]["column"].split(" & ")
        if len(parts) == 2 and parts[0] in df_out.columns and parts[1] in df_out.columns:
            df_out, int_log = create_interaction_feature(df_out, parts[0], parts[1], operation="ratio")
            changelog.extend(int_log)

    df_out = df_out.loc[:, ~df_out.columns.duplicated()].copy()
    changelog.append(f"1-Click AI Feature Engineering complete: synthesized features across {len(df_out.columns)} total columns.")

    return df_out, changelog
