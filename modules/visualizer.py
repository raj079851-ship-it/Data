"""
Visualizer Module
Creates interactive, publication-ready Plotly charts for EDA, distributions,
correlations, time-series, advanced BI (Treemap, Sunburst, Funnel, Waterfall, Radar, Gauge, Pareto),
and geospatial mappings, plus automated AI chart recommendations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any
from scipy import stats

THEME_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Prism


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    nbins: int = 30,
    show_box: bool = True
) -> go.Figure:
    """Renders an interactive distribution histogram with marginal box plot."""
    marginal = "box" if show_box else None
    fig = px.histogram(
        df,
        x=column,
        marginal=marginal,
        nbins=nbins,
        template=THEME_TEMPLATE,
        color_discrete_sequence=["#4f46e5"],
        title=f"Distribution of <b>{column}</b>"
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count / Frequency",
        bargap=0.05,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_categorical_frequency(
    df: pd.DataFrame,
    column: str,
    top_n: int = 12
) -> go.Figure:
    """Renders frequency bar chart for categorical variables."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    counts["Percentage"] = (counts["Count"] / len(df.dropna(subset=[column])) * 100).round(1)

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        text=counts["Percentage"].apply(lambda p: f"{p}%"),
        template=THEME_TEMPLATE,
        color="Count",
        color_continuous_scale="Blues",
        title=f"Frequency Breakdown: <b>{column}</b> (Top {top_n})"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Renders an annotated correlation heatmap with RdBu color scale."""
    if corr_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient numerical features for correlation heatmap", showarrow=False)
        return fig

    z = corr_df.values
    x = list(corr_df.columns)
    y = list(corr_df.index)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            hoverongaps=False,
            text=np.round(z, 2),
            texttemplate="%{text}",
            textfont={"size": 11}
        )
    )
    fig.update_layout(
        title="<b>Correlation Matrix Heatmap</b>",
        template=THEME_TEMPLATE,
        xaxis=dict(tickangle=-45),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    add_trendline: bool = True
) -> go.Figure:
    """Renders an interactive scatter plot with optional regression trendline."""
    trendline = "ols" if add_trendline else None
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline=trendline,
        trendline_color_override="#ef4444",
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"<b>{x_col}</b> vs <b>{y_col}</b>"
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_box_by_group(
    df: pd.DataFrame,
    category_col: str,
    numeric_col: str
) -> go.Figure:
    """Renders side-by-side box plots for numerical distribution across categorical groups."""
    fig = px.box(
        df,
        x=category_col,
        y=numeric_col,
        color=category_col,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        points="outliers",
        title=f"Distribution of <b>{numeric_col}</b> by <b>{category_col}</b>"
    )
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=numeric_col,
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    dimensions: List[str],
    color_col: Optional[str] = None
) -> go.Figure:
    """Renders a multivariate pairwise scatter plot matrix (pairplot)."""
    fig = px.scatter_matrix(
        df,
        dimensions=dimensions,
        color=color_col,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="<b>Multivariate Pairwise Scatter Matrix</b>"
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_dashboard_line_chart(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders a time-series trend line chart."""
    chart_title = title or f"Trend Over Time: <b>{metric_col}</b>"
    fig = px.line(
        df,
        x=date_col,
        y=metric_col,
        template=THEME_TEMPLATE,
        title=chart_title,
        color_discrete_sequence=["#4f46e5"]
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=metric_col,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_dashboard_horizontal_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders a horizontal ranking bar chart."""
    chart_title = title or f"<b>{x_col}</b> by <b>{y_col}</b>"
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        template=THEME_TEMPLATE,
        title=chart_title,
        color=x_col,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_dashboard_column_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    barmode: str = "group",
    title: Optional[str] = None
) -> go.Figure:
    """Renders vertical column / grouped bar chart."""
    chart_title = title or f"<b>{y_col}</b> by <b>{x_col}</b>"
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode=barmode,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=chart_title
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_dashboard_donut_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders a donut composition chart."""
    chart_title = title or f"Composition of <b>{values_col}</b> by <b>{names_col}</b>"
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        hole=0.45,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=chart_title
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    return fig


def plot_dashboard_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders a standard pie chart."""
    chart_title = title or f"Share of <b>{values_col}</b> by <b>{names_col}</b>"
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=chart_title
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    return fig


def plot_dashboard_map_chart(
    df: pd.DataFrame,
    metric_col: str,
    dim_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Renders geospatial scatter/bubble map or regional mapping."""
    chart_title = title or f"Geospatial / Regional Mapping: <b>{metric_col}</b>"

    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    found_lat = lat_col or cols_lower.get("lat") or cols_lower.get("latitude")
    found_lon = lon_col or cols_lower.get("lon") or cols_lower.get("lng") or cols_lower.get("longitude")

    if found_lat and found_lon:
        map_df = df.dropna(subset=[found_lat, found_lon, metric_col]).copy()
        if not map_df.empty:
            map_df = map_df[(map_df[found_lat] >= -90) & (map_df[found_lat] <= 90)]
            map_df = map_df[(map_df[found_lon] >= -180) & (map_df[found_lon] <= 180)]
            if not map_df.empty:
                fig = px.scatter_geo(
                    map_df,
                    lat=found_lat,
                    lon=found_lon,
                    color=metric_col,
                    size=map_df[metric_col].clip(lower=1) if (map_df[metric_col] > 0).any() else None,
                    hover_name=dim_col if dim_col and dim_col in map_df.columns else None,
                    color_continuous_scale="Viridis",
                    template=THEME_TEMPLATE,
                    title=chart_title,
                    projection="natural earth"
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                return fig

    geo_name_cols = [c for c in df.columns if any(k in c.lower() for k in ["country", "state", "region", "city", "location", "neighborhood"])]
    if geo_name_cols:
        loc_col = geo_name_cols[0]
        agg_geo = df.groupby(loc_col)[metric_col].mean().reset_index().head(25)
        fig = px.choropleth(
            agg_geo,
            locations=loc_col,
            locationmode="country names" if "country" in loc_col.lower() else "USA-states",
            color=metric_col,
            color_continuous_scale="Blues",
            template=THEME_TEMPLATE,
            title=f"Regional Choropleth: <b>{metric_col}</b> by <b>{loc_col}</b>"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    # Fallback to pseudo-coordinate bubble projection
    display_dim = dim_col if dim_col and dim_col in df.columns else df.columns[0]
    agg_spatial = df.groupby(display_dim)[metric_col].agg(["count", "mean"]).reset_index().head(20)
    agg_spatial.columns = [display_dim, "RecordCount", "MeanMetric"]
    n = len(agg_spatial)
    lats = [float(20 + 40 * np.sin(2 * np.pi * i / max(n, 1))) for i in range(n)]
    lons = [float(-40 + 160 * np.cos(2 * np.pi * i / max(n, 1))) for i in range(n)]
    agg_spatial["Latitude"] = lats
    agg_spatial["Longitude"] = lons

    fig = px.scatter_geo(
        agg_spatial,
        lat="Latitude",
        lon="Longitude",
        color="MeanMetric",
        size="RecordCount",
        hover_name=display_dim,
        color_continuous_scale="Turbo",
        template=THEME_TEMPLATE,
        projection="natural earth",
        title=f"Spatial Segment Cluster Map: <b>{metric_col}</b> by <b>{display_dim}</b>"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ---------------- NEW ADVANCED STATISTICAL & BI CHARTS ----------------

def plot_violin(
    df: pd.DataFrame,
    category_col: str,
    numeric_col: str,
    box: bool = True
) -> go.Figure:
    """Renders violin plot showing probability density and box elements."""
    fig = px.violin(
        df,
        x=category_col,
        y=numeric_col,
        color=category_col,
        box=box,
        points="outliers",
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Violin Density: <b>{numeric_col}</b> across <b>{category_col}</b>"
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_density_kde(
    df: pd.DataFrame,
    column: str,
    group_col: Optional[str] = None
) -> go.Figure:
    """Renders Kernel Density Estimate (KDE) curve."""
    s = df[column].dropna()
    fig = go.Figure()

    if group_col and group_col in df.columns:
        for val in df[group_col].dropna().unique()[:6]:
            sub = df[df[group_col] == val][column].dropna()
            if len(sub) > 5:
                kde = stats.gaussian_kde(sub)
                x_vals = np.linspace(sub.min(), sub.max(), 150)
                fig.add_trace(go.Scatter(x=x_vals, y=kde(x_vals), mode="lines", name=str(val), fill="tozeroy", opacity=0.4))
        fig.update_layout(title=f"KDE Density Curve: <b>{column}</b> grouped by <b>{group_col}</b>")
    else:
        if len(s) > 5:
            kde = stats.gaussian_kde(s)
            x_vals = np.linspace(s.min(), s.max(), 200)
            fig.add_trace(go.Scatter(x=x_vals, y=kde(x_vals), mode="lines", name="KDE", fill="tozeroy", line=dict(color="#4f46e5", width=2)))
        fig.update_layout(title=f"Kernel Density Estimation (KDE): <b>{column}</b>")

    fig.update_layout(template=THEME_TEMPLATE, xaxis_title=column, yaxis_title="Probability Density", margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_ecdf(df: pd.DataFrame, column: str, color_col: Optional[str] = None) -> go.Figure:
    """Renders Empirical Cumulative Distribution Function (ECDF)."""
    fig = px.ecdf(
        df,
        x=column,
        color=color_col,
        template=THEME_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Empirical Cumulative Distribution Function (ECDF): <b>{column}</b>"
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Cumulative Probability", margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_qq(df: pd.DataFrame, column: str) -> go.Figure:
    """Renders Normal Quantile-Quantile (Q-Q) plot."""
    s = df[column].dropna()
    res = stats.probplot(s, dist="norm")
    theoretical_quantiles = res[0][0]
    sample_quantiles = res[0][1]
    slope, intercept, r = res[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode="markers", name="Observed Quantiles", marker=dict(color="#4f46e5", size=5)))
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=slope * theoretical_quantiles + intercept, mode="lines", name=f"Fit Line (R={round(r, 3)})", line=dict(color="#ef4444", dash="dash")))
    fig.update_layout(
        title=f"Normal Q-Q Plot: <b>{column}</b> (Normality Check)",
        xaxis_title="Theoretical Quantiles (Normal Dist)",
        yaxis_title="Sample Quantiles",
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_treemap(
    df: pd.DataFrame,
    path_cols: List[str],
    values_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders hierarchical Treemap."""
    chart_title = title or f"Treemap: <b>{values_col}</b> by <b>{' > '.join(path_cols)}</b>"
    fig = px.treemap(
        df,
        path=path_cols,
        values=values_col,
        color=values_col,
        color_continuous_scale="Viridis",
        template=THEME_TEMPLATE,
        title=chart_title
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_sunburst(
    df: pd.DataFrame,
    path_cols: List[str],
    values_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """Renders radial Sunburst hierarchical chart."""
    chart_title = title or f"Sunburst: <b>{values_col}</b> by <b>{' > '.join(path_cols)}</b>"
    fig = px.sunburst(
        df,
        path=path_cols,
        values=values_col,
        color=values_col,
        color_continuous_scale="Purples",
        template=THEME_TEMPLATE,
        title=chart_title
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_funnel(
    stages: List[str],
    values: List[float],
    title: str = "Conversion / Sales Funnel"
) -> go.Figure:
    """Renders a business conversion funnel chart."""
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent previous",
        marker=dict(color=["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"])
    ))
    fig.update_layout(title=f"<b>{title}</b>", template=THEME_TEMPLATE, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_waterfall(
    measure: List[str],
    x: List[str],
    y: List[float],
    title: str = "Variance / Contribution Waterfall"
) -> go.Figure:
    """Renders financial / contribution waterfall chart."""
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#4f46e5"}}
    ))
    fig.update_layout(title=f"<b>{title}</b>", template=THEME_TEMPLATE, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_radar(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    title: str = "Multidimensional Radar / Spider Analysis"
) -> go.Figure:
    """Renders radar / spider web chart."""
    fig = go.Figure()
    for name, vals in values_dict.items():
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"<b>{title}</b>",
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_gauge(
    value: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
    title: str = "Performance Gauge",
    target: Optional[float] = None
) -> go.Figure:
    """Renders circular KPI gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if target is not None else "gauge+number",
        value=value,
        delta={"reference": target} if target is not None else None,
        title={"text": f"<b>{title}</b>"},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#4f46e5"},
            "steps": [
                {"range": [min_val, max_val * 0.5], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [max_val * 0.8, max_val], "color": "rgba(16, 185, 129, 0.2)"}
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": target or max_val * 0.9}
        }
    ))
    fig.update_layout(template=THEME_TEMPLATE, margin=dict(l=30, r=30, t=50, b=30), height=320)
    return fig


def plot_pareto(df: pd.DataFrame, category_col: str, value_col: str) -> go.Figure:
    """Renders Pareto 80/20 chart combining bars and cumulative percentage line."""
    agg = df.groupby(category_col)[value_col].sum().sort_values(ascending=False).reset_index()
    agg["Cumulative"] = agg[value_col].cumsum()
    total = agg[value_col].sum()
    agg["Cum_Pct"] = (agg["Cumulative"] / total * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg[category_col], y=agg[value_col], name=value_col, marker_color="#4f46e5"))
    fig.add_trace(go.Scatter(x=agg[category_col], y=agg["Cum_Pct"], name="Cumulative %", yaxis="y2", line=dict(color="#f59e0b", width=3)))

    fig.update_layout(
        title=f"Pareto Analysis (80/20 Rule): <b>{value_col}</b> by <b>{category_col}</b>",
        template=THEME_TEMPLATE,
        yaxis=dict(title=value_col),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        legend=dict(x=0.7, y=1.1, orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def recommend_best_chart(
    df: pd.DataFrame,
    col1: str,
    col2: Optional[str] = None
) -> Dict[str, Any]:
    """
    AI Visualization Recommender:
    Evaluates column types and cardinalities to recommend the optimal chart and rationale.
    """
    if col1 not in df.columns:
        return {"chart_type": "bar", "title": "Default Chart", "rationale": "Column not found"}

    is_num1 = pd.api.types.is_numeric_dtype(df[col1])
    is_date1 = pd.api.types.is_datetime64_any_dtype(df[col1]) or "date" in col1.lower()

    if col2 is None:
        # Single variable
        if is_num1:
            return {
                "chart_type": "distribution",
                "title": f"Distribution Analysis of {col1}",
                "rationale": "Single numeric feature: Histogram with marginal box plot highlights skewness, dispersion, and outliers."
            }
        else:
            return {
                "chart_type": "categorical_frequency",
                "title": f"Frequency Breakdown of {col1}",
                "rationale": "Single categorical feature: Ranked bar chart highlights dominant segments and cardinality."
            }

    # Two variables
    is_num2 = pd.api.types.is_numeric_dtype(df[col2])
    is_date2 = pd.api.types.is_datetime64_any_dtype(df[col2]) or "date" in col2.lower()

    if is_date1 and is_num2:
        return {
            "chart_type": "line",
            "title": f"{col2} Trend Over Time",
            "rationale": "Datetime vs Numeric: Time-series line chart tracks velocity, seasonality, and long-term momentum."
        }
    elif is_num1 and is_num2:
        return {
            "chart_type": "scatter",
            "title": f"Correlation: {col1} vs {col2}",
            "rationale": "Two numeric variables: Scatter plot with OLS trendline reveals linear and non-linear correlation."
        }
    elif not is_num1 and is_num2:
        n_unique = df[col1].nunique()
        if n_unique <= 6:
            return {
                "chart_type": "donut",
                "title": f"{col2} Share by {col1}",
                "rationale": "Low-cardinality category vs numeric: Donut chart displays proportional contribution cleanly."
            }
        else:
            return {
                "chart_type": "box_by_group",
                "title": f"{col2} Spread across {col1}",
                "rationale": "Categorical vs numeric: Grouped box plot compares median, IQR, and variance between segments."
            }
    else:
        return {
            "chart_type": "column",
            "title": f"{col1} by {col2}",
            "rationale": "Categorical relationship: Segment breakdown."
        }
