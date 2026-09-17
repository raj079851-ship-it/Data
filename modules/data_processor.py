"""
Data Processor Module — Comprehensive Visual Transformation Engine
Supports:
- Row filtering (multi-condition AND/OR builders)
- Column selection, renaming, dropping
- Calculated columns (mathematical, logic, string expressions)
- Sorting & multi-level sorting
- Group By & Aggregations (SUM, AVG, MIN, MAX, COUNT, DISTINCT COUNT, MEDIAN, STD, VARIANCE)
- Relational Joins (Inner, Left, Right, Full Outer, Cross)
- Unions, Concatenation & Merges
- Pivoting & Unpivoting (Melt)
- Splitting & Combining columns
- Sampling (random %, stratified, head/tail)
- Ranking (dense_rank, row_number, percent_rank, min_rank)
- Window functions (rolling mean, cumsum, lag, lead, expanding)
- Binning / Discretization (equal-width, quantile, custom edges)
- Resampling (time-series frequency aggregation)
- Date transformations (extract, diff, format, shift)
- String transformations (upper, lower, trim, replace, extract, pad)
- Mathematical transformations (log, sqrt, abs, round, normalize, standardize)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union


def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any
) -> pd.DataFrame:
    """
    Filters rows based on operator:
    '==', '!=', '>', '>=', '<', '<=', 'contains', 'startswith', 'endswith', 'isin', 'isnull', 'notnull'
    """
    if column not in df.columns:
        return df

    s = df[column]
    if operator == "isnull":
        return df[s.isna()]
    elif operator == "notnull":
        return df[s.notna()]

    # Numeric conversion if column is numeric
    if pd.api.types.is_numeric_dtype(s):
        try:
            val_num = float(value)
            if operator == "==":
                return df[s == val_num]
            elif operator == "!=":
                return df[s != val_num]
            elif operator == ">":
                return df[s > val_num]
            elif operator == ">=":
                return df[s >= val_num]
            elif operator == "<":
                return df[s < val_num]
            elif operator == "<=":
                return df[s <= val_num]
        except (ValueError, TypeError):
            pass

    # String operations
    s_str = s.astype(str)
    val_str = str(value)

    if operator == "==":
        return df[s_str == val_str]
    elif operator == "!=":
        return df[s_str != val_str]
    elif operator == "contains":
        return df[s_str.str.contains(val_str, case=False, na=False)]
    elif operator == "startswith":
        return df[s_str.str.startswith(val_str, na=False)]
    elif operator == "endswith":
        return df[s_str.str.endswith(val_str, na=False)]
    elif operator == "isin":
        if isinstance(value, str):
            vals = [v.strip() for v in value.split(",")]
        else:
            vals = list(value)
        return df[s.isin(vals)]

    return df


def select_and_rename_columns(
    df: pd.DataFrame,
    keep_columns: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Selects, renames, and drops columns."""
    out = df.copy()
    if drop_columns:
        out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    if keep_columns:
        valid_keep = [c for c in keep_columns if c in out.columns]
        if valid_keep:
            out = out[valid_keep]
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def create_calculated_column(
    df: pd.DataFrame,
    new_col_name: str,
    expression: str
) -> Tuple[pd.DataFrame, str]:
    """
    Evaluates a Python/Pandas expression to create a new calculated column.
    Example expression: "df['Revenue'] - df['Cost']" or "df['UnitsSold'] * df['UnitPrice']"
    """
    out = df.copy()
    safe_dict = {
        "df": out,
        "np": np,
        "pd": pd
    }
    # Add column variables for easy formula writing, e.g. "Revenue - Cost"
    for col in out.columns:
        clean_name = "".join(c for c in col if c.isalnum() or c == "_")
        if clean_name and not clean_name[0].isdigit():
            safe_dict[clean_name] = out[col]

    try:
        # Check if expression is direct arithmetic
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        out[new_col_name] = result
        return out, f"Created column '{new_col_name}' successfully."
    except Exception as e:
        return df, f"Error calculating column: {str(e)}"


def group_by_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_configs: Dict[str, Union[str, List[str]]]
) -> pd.DataFrame:
    """
    Groups by group_cols and computes aggregates:
    SUM, AVG (mean), MIN, MAX, COUNT, DISTINCT COUNT (nunique), MEDIAN, STD, VARIANCE (var)
    """
    if not group_cols or not agg_configs:
        return df

    func_map = {
        "sum": "sum",
        "avg": "mean",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "count": "count",
        "distinct_count": "nunique",
        "median": "median",
        "std": "std",
        "variance": "var"
    }

    clean_aggs = {}
    for col, funcs in agg_configs.items():
        if col not in df.columns:
            continue
        if isinstance(funcs, str):
            funcs = [funcs]
        mapped = [func_map.get(f.lower(), "mean") for f in funcs]
        clean_aggs[col] = mapped

    if not clean_aggs:
        return df

    grouped = df.groupby(group_cols).agg(clean_aggs)
    
    # Flatten MultiIndex columns if needed
    if isinstance(grouped.columns, pd.MultiIndex):
        col_names = []
        for col, func in grouped.columns:
            req_funcs = [str(f).lower() for f in (agg_configs.get(col) if isinstance(agg_configs.get(col), list) else [agg_configs.get(col, '')])]
            label = "avg" if (func == "mean" and "avg" in req_funcs) else func
            col_names.append(f"{col}_{label}")
        grouped.columns = col_names
    
    return grouped.reset_index()


def join_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner" # 'inner', 'left', 'right', 'outer', 'cross'
) -> pd.DataFrame:
    """Performs relational join between two datasets."""
    if how == "cross":
        return left_df.merge(right_df, how="cross")
    return left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how)


def union_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ignore_index: bool = True
) -> pd.DataFrame:
    """Stacks two datasets vertically (UNION ALL)."""
    return pd.concat([df1, df2], ignore_index=ignore_index)


def pivot_dataset(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean"
) -> pd.DataFrame:
    """Pivots DataFrame into cross-tabulated shape."""
    pivoted = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    pivoted.columns = [f"{col}" for col in pivoted.columns]
    return pivoted.reset_index()


def unpivot_dataset(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: List[str],
    var_name: str = "Variable",
    value_name: str = "Value"
) -> pd.DataFrame:
    """Unpivots (melts) DataFrame from wide to long format."""
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)


def split_combine_columns(
    df: pd.DataFrame,
    action: str, # 'split' or 'combine'
    target_cols: List[str],
    separator: str = " ",
    new_col_name: str = "Combined_Col"
) -> pd.DataFrame:
    """Splits a single column by separator or combines multiple columns into one."""
    out = df.copy()
    if action == "split" and len(target_cols) == 1:
        col = target_cols[0]
        splits = out[col].astype(str).str.split(separator, expand=True)
        for i in range(splits.shape[1]):
            out[f"{col}_part_{i+1}"] = splits[i]
    elif action == "combine" and len(target_cols) > 1:
        combined_series = out[target_cols[0]].astype(str)
        for c in target_cols[1:]:
            combined_series = combined_series + separator + out[c].astype(str)
        out[new_col_name] = combined_series
    return out


# ============================================================
# NEW: ENHANCED TRANSFORMATION FUNCTIONS
# ============================================================

def sort_dataframe(
    df: pd.DataFrame,
    sort_configs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Multi-level sort.
    sort_configs: [{"column": "Revenue", "ascending": False}, ...]
    """
    if not sort_configs:
        return df
    cols = [c["column"] for c in sort_configs if c["column"] in df.columns]
    asc = [c.get("ascending", True) for c in sort_configs if c["column"] in df.columns]
    if not cols:
        return df
    return df.sort_values(by=cols, ascending=asc).reset_index(drop=True)


def sample_dataframe(
    df: pd.DataFrame,
    method: str = "random_pct",
    value: float = 10.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Samples rows from the DataFrame.
    method: 'random_pct' (value=%), 'random_n' (value=n rows), 'head' (first n), 'tail' (last n)
    """
    n = len(df)
    if n == 0:
        return df
    if method == "random_pct":
        frac = min(max(float(value) / 100.0, 0.0), 1.0)
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    elif method == "random_n":
        k = min(int(value), n)
        return df.sample(n=k, random_state=seed).reset_index(drop=True)
    elif method == "head":
        return df.head(int(value)).reset_index(drop=True)
    elif method == "tail":
        return df.tail(int(value)).reset_index(drop=True)
    return df


def rank_column(
    df: pd.DataFrame,
    column: str,
    method: str = "dense",  # 'dense', 'first', 'min', 'max', 'average'
    ascending: bool = False,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Adds a ranking column based on the specified column.
    method maps to pandas rank method: dense=dense_rank, first=row_number, min=min_rank.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    rank_col = new_col_name or f"{column}_rank"
    out[rank_col] = out[column].rank(method=method, ascending=ascending).astype(int)
    return out


def window_function(
    df: pd.DataFrame,
    column: str,
    func: str,  # 'rolling_mean', 'rolling_sum', 'cumsum', 'lag', 'lead', 'expanding_mean'
    window: int = 3,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Applies window / analytic functions to a numeric column.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    s = out[column]
    col_label = new_col_name or f"{column}_{func}"
    if func == "rolling_mean":
        out[col_label] = s.rolling(window=window, min_periods=1).mean()
    elif func == "rolling_sum":
        out[col_label] = s.rolling(window=window, min_periods=1).sum()
    elif func == "rolling_min":
        out[col_label] = s.rolling(window=window, min_periods=1).min()
    elif func == "rolling_max":
        out[col_label] = s.rolling(window=window, min_periods=1).max()
    elif func == "cumsum":
        out[col_label] = s.cumsum()
    elif func == "cumprod":
        out[col_label] = s.cumprod()
    elif func == "expanding_mean":
        out[col_label] = s.expanding(min_periods=1).mean()
    elif func == "lag":
        out[col_label] = s.shift(window)
    elif func == "lead":
        out[col_label] = s.shift(-window)
    elif func == "pct_change":
        out[col_label] = s.pct_change(periods=window).round(4)
    return out


def bin_column(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",  # 'equal_width', 'quantile', 'custom'
    bins: int = 5,
    labels: Optional[List[str]] = None,
    custom_edges: Optional[List[float]] = None,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Discretizes / bins a numeric column into categories.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    col_label = new_col_name or f"{column}_bin"
    try:
        if method == "equal_width":
            out[col_label] = pd.cut(out[column], bins=bins, labels=labels, duplicates="drop")
        elif method == "quantile":
            out[col_label] = pd.qcut(out[column], q=bins, labels=labels, duplicates="drop")
        elif method == "custom" and custom_edges:
            out[col_label] = pd.cut(out[column], bins=custom_edges, labels=labels, include_lowest=True)
        out[col_label] = out[col_label].astype(str)
    except Exception:
        pass
    return out


def resample_timeseries(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "M",  # 'D', 'W', 'M', 'Q', 'Y'
    agg_func: str = "sum"
) -> pd.DataFrame:
    """
    Resamples a time-series DataFrame to a target frequency.
    Requires a parseable date column.
    """
    if date_column not in df.columns or value_column not in df.columns:
        return df
    out = df.copy()
    try:
        out[date_column] = pd.to_datetime(out[date_column], infer_datetime_format=True)
        out = out.set_index(date_column)
        resampled = out[[value_column]].resample(frequency).agg(agg_func)
        return resampled.reset_index()
    except Exception:
        return df


def date_transform(
    df: pd.DataFrame,
    column: str,
    operation: str,  # 'extract_year', 'extract_month', 'extract_day', 'extract_weekday',
                     # 'extract_hour', 'extract_quarter', 'date_diff_today', 'to_string'
    new_col_name: Optional[str] = None,
    date_format: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """
    Applies date / datetime transformations to a column.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    col_label = new_col_name or f"{column}_{operation}"
    try:
        dt = pd.to_datetime(out[column], infer_datetime_format=True, errors="coerce")
        if operation == "extract_year":
            out[col_label] = dt.dt.year
        elif operation == "extract_month":
            out[col_label] = dt.dt.month
        elif operation == "extract_day":
            out[col_label] = dt.dt.day
        elif operation == "extract_weekday":
            out[col_label] = dt.dt.day_name()
        elif operation == "extract_hour":
            out[col_label] = dt.dt.hour
        elif operation == "extract_quarter":
            out[col_label] = dt.dt.quarter
        elif operation == "extract_week":
            out[col_label] = dt.dt.isocalendar().week.astype(int)
        elif operation == "date_diff_today":
            out[col_label] = (pd.Timestamp.now() - dt).dt.days
        elif operation == "to_string":
            out[col_label] = dt.dt.strftime(date_format)
        elif operation == "is_weekend":
            out[col_label] = dt.dt.dayofweek >= 5
        elif operation == "month_name":
            out[col_label] = dt.dt.month_name()
    except Exception:
        pass
    return out


def string_transform(
    df: pd.DataFrame,
    column: str,
    operation: str,  # 'upper', 'lower', 'title', 'strip', 'replace', 'extract_regex',
                     # 'pad_left', 'pad_right', 'length', 'contains_flag', 'remove_special'
    new_col_name: Optional[str] = None,
    pattern: str = "",
    replacement: str = "",
    width: int = 10
) -> pd.DataFrame:
    """
    Applies string transformations to a text column.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    col_label = new_col_name or f"{column}_{operation}"
    s = out[column].astype(str)
    try:
        if operation == "upper":
            out[col_label] = s.str.upper()
        elif operation == "lower":
            out[col_label] = s.str.lower()
        elif operation == "title":
            out[col_label] = s.str.title()
        elif operation == "strip":
            out[col_label] = s.str.strip()
        elif operation == "replace":
            out[col_label] = s.str.replace(pattern, replacement, regex=False)
        elif operation == "extract_regex":
            extracted = s.str.extract(f"({pattern})", expand=False)
            out[col_label] = extracted.fillna("")
        elif operation == "pad_left":
            out[col_label] = s.str.zfill(width)
        elif operation == "pad_right":
            out[col_label] = s.str.ljust(width)
        elif operation == "length":
            out[col_label] = s.str.len()
        elif operation == "contains_flag":
            out[col_label] = s.str.contains(pattern, case=False, na=False)
        elif operation == "remove_special":
            out[col_label] = s.str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
        elif operation == "split_first":
            out[col_label] = s.str.split(pattern, n=1, expand=True)[0]
        elif operation == "split_last":
            parts = s.str.rsplit(pattern, n=1, expand=True)
            out[col_label] = parts[parts.columns[-1]]
    except Exception:
        pass
    return out


def math_transform(
    df: pd.DataFrame,
    column: str,
    operation: str,  # 'log', 'log10', 'sqrt', 'abs', 'square', 'round', 'ceiling', 'floor',
                     # 'normalize', 'standardize', 'invert', 'exp'
    new_col_name: Optional[str] = None,
    decimals: int = 2
) -> pd.DataFrame:
    """
    Applies mathematical transformations to a numeric column.
    """
    if column not in df.columns:
        return df
    out = df.copy()
    col_label = new_col_name or f"{column}_{operation}"
    s = pd.to_numeric(out[column], errors="coerce")
    try:
        if operation == "log":
            out[col_label] = np.log(s.clip(lower=1e-9))
        elif operation == "log10":
            out[col_label] = np.log10(s.clip(lower=1e-9))
        elif operation == "sqrt":
            out[col_label] = np.sqrt(s.clip(lower=0))
        elif operation == "abs":
            out[col_label] = s.abs()
        elif operation == "square":
            out[col_label] = s ** 2
        elif operation == "round":
            out[col_label] = s.round(decimals)
        elif operation == "ceiling":
            out[col_label] = np.ceil(s)
        elif operation == "floor":
            out[col_label] = np.floor(s)
        elif operation == "normalize":
            s_min, s_max = s.min(), s.max()
            out[col_label] = ((s - s_min) / (s_max - s_min + 1e-9)).round(4)
        elif operation == "standardize":
            out[col_label] = ((s - s.mean()) / (s.std() + 1e-9)).round(4)
        elif operation == "invert":
            out[col_label] = 1.0 / (s + 1e-9)
        elif operation == "exp":
            out[col_label] = np.exp(s)
        elif operation == "pct_of_total":
            out[col_label] = (s / s.sum() * 100).round(4)
        elif operation == "cumulative_pct":
            out[col_label] = (s.cumsum() / s.sum() * 100).round(4)
    except Exception:
        pass
    return out


def concatenate_datasets(
    dfs: List[pd.DataFrame],
    axis: int = 0,
    ignore_index: bool = True,
    join: str = "outer"
) -> pd.DataFrame:
    """
    Concatenates multiple DataFrames vertically (axis=0) or horizontally (axis=1).
    """
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index, join=join)


def multi_filter(
    df: pd.DataFrame,
    conditions: List[Dict[str, Any]],
    logic: str = "AND"  # 'AND' or 'OR'
) -> pd.DataFrame:
    """
    Applies multiple filter conditions with AND / OR logic.
    conditions: [{"column": "Revenue", "operator": ">", "value": 1000}, ...]
    """
    if not conditions:
        return df
    masks = []
    for cond in conditions:
        col = cond.get("column")
        op = cond.get("operator", "==")
        val = cond.get("value", "")
        if col not in df.columns:
            continue
        filtered = filter_rows(df, col, op, val)
        mask = df.index.isin(filtered.index)
        masks.append(mask)

    if not masks:
        return df

    import functools
    if logic == "AND":
        combined = functools.reduce(lambda a, b: a & b, masks)
    else:
        combined = functools.reduce(lambda a, b: a | b, masks)
    return df[combined].reset_index(drop=True)

