"""
Data Loader Module
Handles multi-format data ingestion (CSV, XLSX, XLS, JSON, Parquet, TXT, TSV),
automated delimiter & encoding detection, semantic column inference,
Dataset Health Score calculation (0-100), and demo dataset generation.
"""

import io
import os
import json
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def load_dataset(
    file_or_buffer: Any,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads data from uploaded buffer or file path across multiple file formats:
    CSV, XLSX, XLS, JSON, Parquet, TXT, TSV.
    Returns (DataFrame, metadata_dict).
    """
    name = filename or getattr(file_or_buffer, "name", "")
    ext = (file_type or os.path.splitext(name)[1].lower().replace(".", "")).strip().lower()

    raw_bytes = None
    if hasattr(file_or_buffer, "getvalue"):
        val = file_or_buffer.getvalue()
        raw_bytes = val.encode("utf-8") if isinstance(val, str) else val
    elif hasattr(file_or_buffer, "read"):
        val = file_or_buffer.read()
        raw_bytes = val.encode("utf-8") if isinstance(val, str) else val
    elif isinstance(file_or_buffer, (str, bytes, os.PathLike)):
        if isinstance(file_or_buffer, (str, os.PathLike)) and os.path.exists(file_or_buffer):
            with open(file_or_buffer, "rb") as f:
                raw_bytes = f.read()

    df = None
    used_encoding = encoding or "utf-8"
    used_delimiter = delimiter or ","

    try:
        if ext in ["xlsx", "xls"]:
            buffer = io.BytesIO(raw_bytes) if raw_bytes else file_or_buffer
            df = pd.read_excel(buffer, engine="openpyxl" if ext == "xlsx" else None)
        elif ext == "parquet":
            buffer = io.BytesIO(raw_bytes) if raw_bytes else file_or_buffer
            df = pd.read_parquet(buffer)
        elif ext == "json":
            buffer = io.BytesIO(raw_bytes) if raw_bytes else file_or_buffer
            try:
                df = pd.read_json(buffer)
            except Exception:
                if raw_bytes:
                    data = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
                    df = pd.json_normalize(data)
                else:
                    raise
        elif ext in ["tsv", "tab"]:
            return load_csv_data(file_or_buffer, delimiter="\t", encoding=encoding)
        else:
            return load_csv_data(file_or_buffer, delimiter=delimiter, encoding=encoding)
    except Exception as e:
        try:
            return load_csv_data(file_or_buffer, delimiter=delimiter, encoding=encoding)
        except Exception:
            raise ValueError(f"Failed to load dataset: {str(e)}")

    if df is None or df.empty:
        raise ValueError("Dataset is empty or could not be parsed.")

    df.columns = [str(c).strip() for c in df.columns]

    meta = get_dataset_quick_profile(df)
    meta["encoding"] = used_encoding
    meta["delimiter"] = used_delimiter
    meta["format"] = ext
    return df, meta


def load_csv_data(
    file_or_buffer: Any,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads CSV / delimited text data with automated fallback for encoding & delimiters.
    """
    encodings_to_try = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    encodings_to_try = [e for e in encodings_to_try if e]

    delimiters_to_try = [delimiter] if delimiter and delimiter != "auto" else [",", ";", "\t", "|"]
    
    df = None
    used_encoding = "utf-8"
    used_delimiter = ","
    error_msg = ""

    raw_bytes = None
    if hasattr(file_or_buffer, "getvalue"):
        val = file_or_buffer.getvalue()
        raw_bytes = val.encode("utf-8") if isinstance(val, str) else val
    elif hasattr(file_or_buffer, "read"):
        val = file_or_buffer.read()
        raw_bytes = val.encode("utf-8") if isinstance(val, str) else val
    elif isinstance(file_or_buffer, (str, bytes, os.PathLike)):
        if isinstance(file_or_buffer, (str, os.PathLike)) and os.path.exists(file_or_buffer):
            with open(file_or_buffer, "rb") as f:
                raw_bytes = f.read()

    for enc in encodings_to_try:
        for sep in delimiters_to_try:
            try:
                if raw_bytes is not None:
                    buffer = io.BytesIO(raw_bytes)
                    df = pd.read_csv(buffer, sep=sep, encoding=enc, engine="c", on_bad_lines="skip")
                else:
                    df = pd.read_csv(file_or_buffer, sep=sep, encoding=enc, engine="c", on_bad_lines="skip")
                
                if len(df.columns) > 1 or (len(df.columns) == 1 and sep == ","):
                    used_encoding = enc
                    used_delimiter = sep
                    break
            except Exception as e:
                error_msg = str(e)
                continue
        if df is not None and (len(df.columns) > 1 or (len(df.columns) == 1 and used_delimiter == ",")):
            break

    if df is None:
        raise ValueError(f"Could not read CSV file. Last error: {error_msg}")

    df.columns = [str(c).strip() for c in df.columns]
    meta = get_dataset_quick_profile(df)
    meta["encoding"] = used_encoding
    meta["delimiter"] = used_delimiter
    meta["format"] = "csv"
    return df, meta


def infer_column_metadata(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infers semantic data types, statistical profiles, and analytical roles
    for all columns in a DataFrame.
    """
    col_meta = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / n_rows * 100), 2) if n_rows > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))
        unique_pct = round((n_unique / n_rows * 100), 2) if n_rows > 0 else 0.0

        col_lower = str(col).lower()
        role = "text"
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        if not is_datetime and series.dropna().shape[0] > 0:
            sample_val = str(series.dropna().iloc[0])
            if re.search(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", sample_val) or any(
                w in col_lower for w in ["date", "timestamp", "time", "created_at", "updated_at", "signup"]
            ):
                try:
                    pd.to_datetime(series.dropna().iloc[:30], errors="raise")
                    is_datetime = True
                except Exception:
                    pass

        if is_bool:
            role = "boolean"
        elif is_datetime:
            role = "datetime"
        elif is_numeric:
            if n_unique <= 2:
                role = "boolean"
            elif any(id_kw in col_lower for id_kw in ["_id", "id", "code", "zip", "postal"]) and unique_pct > 80:
                role = "id"
            elif n_unique < 12 and unique_pct < 5:
                role = "categorical"
            else:
                role = "numeric"
        else:
            if any(id_kw in col_lower for id_kw in ["id", "uuid", "guid", "key", "code"]) and unique_pct > 70:
                role = "id"
            elif n_unique <= 20 or (unique_pct < 10 and n_unique < 50):
                role = "categorical"
            else:
                role = "text"

        is_target_candidate = False
        target_keywords = ["target", "label", "churn", "price", "revenue", "sales", "converted", "fraud", "outcome", "status", "class"]
        if any(kw == col_lower or kw in col_lower for kw in target_keywords):
            is_target_candidate = True

        stats = {
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": n_unique,
            "unique_pct": unique_pct,
            "role": role,
            "dtype": str(series.dtype),
            "is_target_candidate": is_target_candidate
        }

        if is_numeric:
            non_null = series.dropna()
            if len(non_null) > 0:
                stats["min"] = float(non_null.min())
                stats["max"] = float(non_null.max())
                stats["mean"] = float(round(non_null.mean(), 4))
                stats["median"] = float(round(non_null.median(), 4))
                stats["std"] = float(round(non_null.std(), 4)) if len(non_null) > 1 else 0.0
                q25, q75 = non_null.quantile(0.25), non_null.quantile(0.75)
                stats["q25"] = float(round(q25, 4))
                stats["q75"] = float(round(q75, 4))
                iqr = q75 - q25
                outliers = non_null[(non_null < (q25 - 1.5 * iqr)) | (non_null > (q75 + 1.5 * iqr))]
                stats["outlier_count"] = int(len(outliers))
            else:
                stats.update({"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "outlier_count": 0})
        elif role == "categorical":
            vc = series.value_counts(dropna=True)
            stats["top_category"] = str(vc.index[0]) if len(vc) > 0 else ""
            stats["top_category_freq"] = int(vc.iloc[0]) if len(vc) > 0 else 0

        col_meta[col] = stats

    return col_meta


def compute_dataset_health_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes an automated Dataset Health Score from 0 to 100 based on:
    - Missing data penalty (up to -35)
    - Duplicate rows penalty (up to -20)
    - Constant / zero-variance columns (up to -15)
    - Extreme outlier incidence (up to -15)
    """
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] > 0 and df.shape[1] > 0 else 0
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicated_rows = int(df.duplicated().sum())
    dup_pct = round((duplicated_rows / len(df) * 100), 2) if len(df) > 0 else 0.0

    score = 100.0
    issues = []

    if missing_pct > 0:
        pen = min(35.0, missing_pct * 1.5)
        score -= pen
        if missing_pct > 5:
            issues.append(f"{missing_pct}% of data cells are missing ({missing_cells:,} cells)")

    if dup_pct > 0:
        pen = min(20.0, dup_pct * 2.0)
        score -= pen
        issues.append(f"{duplicated_rows:,} duplicate rows ({dup_pct}%)")

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        pen = min(15.0, len(constant_cols) * 5.0)
        score -= pen
        issues.append(f"{len(constant_cols)} constant column(s) with zero variance: {', '.join(constant_cols)}")

    num_df = df.select_dtypes(include=[np.number])
    outlier_cells = 0
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) > 10:
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                outlier_cells += len(s[(s < (q25 - 2.5 * iqr)) | (s > (q75 + 2.5 * iqr))])
    outlier_pct = round((outlier_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    if outlier_pct > 0.5:
        pen = min(15.0, outlier_pct * 2.5)
        score -= pen
        issues.append(f"High extreme outlier incidence: ~{outlier_cells:,} values ({outlier_pct}%)")

    final_score = int(np.clip(round(score), 0, 100))
    if final_score >= 90:
        grade = "A"
        status = "Excellent"
        badge_class = "badge-success"
    elif final_score >= 75:
        grade = "B"
        status = "Good"
        badge_class = "badge-info"
    elif final_score >= 60:
        grade = "C"
        status = "Fair (Needs Cleaning)"
        badge_class = "badge-warning"
    else:
        grade = "D"
        status = "Poor (Critical Cleaning Required)"
        badge_class = "badge-danger"

    return {
        "health_score": final_score,
        "grade": grade,
        "status": status,
        "badge_class": badge_class,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicated_rows": duplicated_rows,
        "duplicated_pct": dup_pct,
        "constant_columns": constant_cols,
        "outlier_cells": outlier_cells,
        "outlier_pct": outlier_pct,
        "issues": issues if issues else ["No major data hygiene anomalies detected."]
    }


def get_dataset_quick_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive high-level profile statistics for a DataFrame."""
    health = compute_dataset_health_score(df)
    col_metadata = infer_column_metadata(df)

    numeric_cols = [c for c, m in col_metadata.items() if m["role"] == "numeric"]
    categorical_cols = [c for c, m in col_metadata.items() if m["role"] == "categorical"]
    datetime_cols = [c for c, m in col_metadata.items() if m["role"] == "datetime"]
    bool_cols = [c for c, m in col_metadata.items() if m["role"] == "boolean"]
    text_cols = [c for c, m in col_metadata.items() if m["role"] == "text"]
    id_cols = [c for c, m in col_metadata.items() if m["role"] == "id"]
    target_cols = [c for c, m in col_metadata.items() if m["is_target_candidate"]]

    total_cells = df.shape[0] * df.shape[1] if df.shape[0] > 0 and df.shape[1] > 0 else 0
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_cells": total_cells,
        "missing_cells": health["missing_cells"],
        "missing_pct": health["missing_pct"],
        "duplicated_rows": health["duplicated_rows"],
        "duplicated_pct": health["duplicated_pct"],
        "health_score": health["health_score"],
        "health_grade": health["grade"],
        "health_status": health["status"],
        "health_issues": health["issues"],
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "datetime_cols_count": len(datetime_cols),
        "bool_cols_count": len(bool_cols),
        "text_cols_count": len(text_cols),
        "id_cols_count": len(id_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "bool_cols": bool_cols,
        "text_cols": text_cols,
        "id_cols": id_cols,
        "target_candidates": target_cols,
        "memory_mb": memory_usage_mb,
        "column_names": list(df.columns),
        "columns_metadata": col_metadata
    }


def generate_sample_ecommerce_sales() -> pd.DataFrame:
    """Generates an E-Commerce Sales dataset ideal for forecasting, SQL, BI, and ML."""
    np.random.seed(101)
    n = 800
    order_ids = [f"ORD-{50000 + i}" for i in range(n)]
    regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    categories = np.random.choice(["Electronics", "Home & Office", "Apparel", "Beauty & Care"], size=n, p=[0.35, 0.3, 0.2, 0.15])
    
    product_map = {
        "Electronics": ["Pro Laptop", "Noise-Cancelling Headphones", "Ultra HD Monitor", "Smart Watch"],
        "Home & Office": ["Ergonomic Chair", "Standing Desk", "Coffee Maker", "Desk Lamp"],
        "Apparel": ["Performance Jacket", "Merino Sweater", "Trail Shoes", "Classic Tee"],
        "Beauty & Care": ["Serum Pack", "Face Cleanser", "Hydration Cream", "Mineral Sunscreen"]
    }
    products = [np.random.choice(product_map[cat]) for cat in categories]
    
    units = np.random.choice([1, 2, 3, 4, 5, 8, 10], size=n, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01])
    base_unit_prices = {
        "Pro Laptop": 1200.0, "Noise-Cancelling Headphones": 250.0, "Ultra HD Monitor": 450.0, "Smart Watch": 299.0,
        "Ergonomic Chair": 350.0, "Standing Desk": 600.0, "Coffee Maker": 180.0, "Desk Lamp": 65.0,
        "Performance Jacket": 160.0, "Merino Sweater": 110.0, "Trail Shoes": 130.0, "Classic Tee": 35.0,
        "Serum Pack": 85.0, "Face Cleanser": 28.0, "Hydration Cream": 45.0, "Mineral Sunscreen": 32.0
    }
    unit_price = np.array([base_unit_prices[p] * np.random.uniform(0.95, 1.05) for p in products])
    revenue = np.round(unit_price * units, 2)
    
    margin_rate = np.random.uniform(0.20, 0.55, size=n)
    cost = np.round(revenue * (1.0 - margin_rate), 2)
    profit = np.round(revenue - cost, 2)
    
    date_range = pd.date_range(start="2022-01-01", end="2024-06-30", periods=n)
    order_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in np.random.choice(date_range, size=n, replace=True)]
    order_dates.sort()

    customer_segments = np.random.choice(["Enterprise", "SMB", "Consumer"], size=n, p=[0.25, 0.35, 0.40])
    channels = np.random.choice(["Direct Web", "Organic Search", "Paid Ads", "Affiliate"], size=n, p=[0.3, 0.3, 0.25, 0.15])
    satisfaction_score = np.clip(np.round(np.random.normal(4.2, 0.8, size=n), 1), 1.0, 5.0)

    df = pd.DataFrame({
        "OrderID": order_ids,
        "OrderDate": order_dates,
        "Region": regions,
        "Category": categories,
        "Product": products,
        "CustomerSegment": customer_segments,
        "SalesChannel": channels,
        "UnitsSold": units,
        "UnitPrice": np.round(unit_price, 2),
        "Revenue": revenue,
        "Cost": cost,
        "Profit": profit,
        "CustomerSatisfaction": satisfaction_score
    })

    df.loc[np.random.rand(n) < 0.03, "CustomerSatisfaction"] = np.nan
    df.loc[np.random.rand(n) < 0.02, "Cost"] = np.nan

    return df


def generate_sample_customer_churn() -> pd.DataFrame:
    """Generates a realistic customer churn dataset with missing values, noise, and dates."""
    np.random.seed(42)
    n = 600
    customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
    genders = np.random.choice(["Male", "Female", "Prefer not to say"], size=n, p=[0.48, 0.48, 0.04])
    senior_citizen = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    partner = np.random.choice(["Yes", "No"], size=n, p=[0.52, 0.48])
    tenure_months = np.random.randint(1, 72, size=n)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    monthly_charges = np.round(np.random.normal(65.0, 28.0, size=n).clip(18.0, 125.0), 2)
    total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 30, size=n), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    base_date = pd.Timestamp("2024-01-01")
    signup_dates = [base_date - pd.Timedelta(days=int(t * 30.4)) for t in tenure_months]
    
    churn_score = (
        (contract_type == "Month-to-month") * 0.4
        + (monthly_charges > 70) * 0.25
        - (tenure_months > 24) * 0.35
        + np.random.normal(0, 0.15, size=n)
    )
    churn = ["Yes" if s > 0.2 else "No" for s in churn_score]

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "SignupDate": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "Gender": genders,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "TenureMonths": tenure_months,
        "Contract": contract_type,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    mask_miss_monthly = np.random.rand(n) < 0.05
    df.loc[mask_miss_monthly, "MonthlyCharges"] = np.nan
    mask_miss_partner = np.random.rand(n) < 0.04
    df.loc[mask_miss_partner, "Partner"] = np.nan
    mask_miss_total = np.random.rand(n) < 0.06
    df.loc[mask_miss_total, "TotalCharges"] = np.nan

    dup_indices = np.random.choice(n, size=8, replace=False)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)
    df.loc[0:15, "PaymentMethod"] = df.loc[0:15, "PaymentMethod"].apply(lambda x: f"  {x} " if pd.notna(x) else x)
    return df


def generate_sample_housing() -> pd.DataFrame:
    """Generates a housing dataset with realistic numerical and categorical features."""
    np.random.seed(99)
    n = 500
    house_id = [f"PROP-{20000 + i}" for i in range(n)]
    neighborhoods = np.random.choice(["Downtown", "Suburbs", "North Hills", "West End", "Riverside"], size=n)
    sqft_living = np.random.randint(750, 4500, size=n)
    bedrooms = np.clip(np.round(sqft_living / 650 + np.random.normal(0, 0.6, size=n)).astype(int), 1, 6)
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.3, size=n), 1), 1.0, 5.0)
    year_built = np.random.randint(1960, 2024, size=n)
    has_garage = np.random.choice(["Yes", "No"], size=n, p=[0.75, 0.25])
    garage_cars = [int(np.random.choice([1, 2, 3])) if g == "Yes" else 0 for g in has_garage]
    lot_size_acres = np.round(np.random.exponential(scale=0.35, size=n) + 0.1, 2)
    
    price = (
        sqft_living * 195
        + bedrooms * 12000
        + bathrooms * 18000
        + (2024 - year_built) * -450
        + (has_garage == "Yes") * 22000
        + np.random.normal(50000, 35000, size=n)
    )
    price = np.round(np.maximum(price, 85000), -2)

    sale_dates = [
        pd.Timestamp("2024-01-01") - pd.Timedelta(days=int(d))
        for d in np.random.randint(0, 730, size=n)
    ]

    df = pd.DataFrame({
        "PropertyID": house_id,
        "SaleDate": [d.strftime("%Y-%m-%d") for d in sale_dates],
        "Neighborhood": neighborhoods,
        "SqftLiving": sqft_living,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "YearBuilt": year_built,
        "Garage": has_garage,
        "GarageCars": garage_cars,
        "LotSizeAcres": lot_size_acres,
        "SalePrice": price
    })

    df.loc[np.random.rand(n) < 0.05, "LotSizeAcres"] = np.nan
    df.loc[np.random.rand(n) < 0.04, "GarageCars"] = np.nan

    outlier_idx = np.random.choice(n, size=4, replace=False)
    df.loc[outlier_idx[0], "SqftLiving"] = 18500
    df.loc[outlier_idx[1], "SalePrice"] = 4200000

    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


def ensure_sample_data_files():
    """Generates sample datasets and persists them to disk."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    churn_path = os.path.join(SAMPLE_DIR, "customer_churn.csv")
    housing_path = os.path.join(SAMPLE_DIR, "housing_market.csv")
    sales_path = os.path.join(SAMPLE_DIR, "ecommerce_sales.csv")

    if not os.path.exists(churn_path):
        df_churn = generate_sample_customer_churn()
        df_churn.to_csv(churn_path, index=False)

    if not os.path.exists(housing_path):
        df_housing = generate_sample_housing()
        df_housing.to_csv(housing_path, index=False)

    if not os.path.exists(sales_path):
        df_sales = generate_sample_ecommerce_sales()
        df_sales.to_csv(sales_path, index=False)
