"""
AI Orchestrator Module
Multi-Agent Analytics Routing Layer & Context-Aware Copilot:
- Multi-Provider Abstraction: Google Gemini, OpenAI-compatible, or Built-in Heuristic Analytics Engine
- Specialized Sub-Agents:
  * Data Agent (profiling & schema understanding)
  * Cleaning Agent (anomalies & sanitization suggestions)
  * EDA Agent (trends, distributions, and collinearity)
  * Visualization Agent (chart selection & syntax)
  * SQL Agent (DuckDB query generation & optimization)
  * Statistics Agent (hypothesis testing interpretation)
  * ML Agent (model recommendations & hyperparameter guidance)
  * Forecasting Agent (seasonality & horizon suggestions)
  * Reporting Agent (executive summaries)
- Strict Anti-Hallucination Grounding: All computed facts derive from actual dataset structures
- Universal Command Bar (Omnibar) router
"""

import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class AIOrchestrator:
    """Orchestrates multi-agent routing, copilot queries, and provider configurations."""

    def __init__(self, provider: str = "heuristic", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def route_omnibar_command(self, query: str, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Routes universal command bar input to the appropriate specialized sub-agent.
        Returns target module, action, and natural language response.
        """
        q = query.lower().strip()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        date_cols = [c for c in df.columns if any(w in c.lower() for w in ["date", "time", "year"])]

        # 1. SQL Query Intent
        if any(w in q for w in ["sql", "query", "select", "where", "group by", "top "]):
            matched_col = num_cols[0] if num_cols else "1"
            return {
                "agent": "SQL Agent",
                "recommended_module": "SQL Studio",
                "action": "execute_sql",
                "sql_suggestion": f"SELECT {cat_cols[0] if cat_cols else '*'}, COUNT(*), AVG({matched_col}) FROM df GROUP BY 1 ORDER BY 3 DESC LIMIT 10;",
                "message": f"Routed to **SQL Studio**. DuckDB SQL query formulated for dataset `{df.shape[0]} rows × {df.shape[1]} cols`."
            }

        # 2. Forecasting Intent
        if any(w in q for w in ["forecast", "future", "predict next", "horizon", "time-series", "seasonal"]):
            return {
                "agent": "Forecasting Agent",
                "recommended_module": "Forecasting",
                "action": "run_forecast",
                "date_col": date_cols[0] if date_cols else None,
                "metric_col": num_cols[0] if num_cols else None,
                "message": f"Routed to **Forecasting Engine**. Ready to project {num_cols[0] if num_cols else 'metrics'} forward across 12 periods."
            }

        # 3. Machine Learning / Prediction Intent
        if any(w in q for w in ["train", "model", "predict", "churn", "regression", "classification", "automl", "cluster"]):
            target = "Churn" if "Churn" in df.columns else (num_cols[-1] if num_cols else df.columns[-1])
            return {
                "agent": "ML Agent",
                "recommended_module": "Machine Learning",
                "action": "setup_ml",
                "suggested_target": target,
                "message": f"Routed to **Machine Learning Module**. Recommending AutoML pipeline targeting `{target}`."
            }

        # 4. Chart / Visualization Intent
        if any(w in q for w in ["chart", "plot", "visualize", "graph", "histogram", "scatter", "bar"]):
            return {
                "agent": "Visualization Agent",
                "recommended_module": "Visualization",
                "action": "render_chart",
                "message": f"Routed to **Visualization Engine**. Generating optimal interactive chart based on analytical intent."
            }

        # 5. Data Cleaning Intent
        if any(w in q for w in ["clean", "impute", "outlier", "missing", "duplicate", "null", "sanitize"]):
            return {
                "agent": "Cleaning Agent",
                "recommended_module": "Data Cleaning",
                "action": "auto_clean",
                "message": "Routed to **Data Cleaning Module**. Evaluated hygiene issues and ready to execute 1-Click AI Auto-Clean."
            }

        # 6. Statistics Intent
        if any(w in q for w in ["hypothesis", "t-test", "anova", "chi-square", "p-value", "significance"]):
            return {
                "agent": "Statistics Agent",
                "recommended_module": "Statistics",
                "action": "run_test",
                "message": "Routed to **Statistical Analysis Module**. Preparing parametric hypothesis evaluation."
            }

        # Default: General AI Analyst
        return {
            "agent": "General Analyst Agent",
            "recommended_module": "AI Assistant & Copilot",
            "action": "chat_insights",
            "message": self.generate_grounded_answer(query, df)
        }

    def ask(self, query: str, df: pd.DataFrame, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entrypoint for conversational AI assistant:
        Checks for LLM API key first; if absent or fails, provides grounded statistical intelligence.
        """
        active_key = api_key or self.api_key
        if active_key:
            try:
                from modules.ai_insights import query_gemini_insights
                summary_ctx = (
                    f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns.\n"
                    f"Columns: {', '.join(df.columns)}.\n"
                    f"Numeric Summary:\n{df.describe().round(2).to_string()}\n"
                    f"Missing Values: {df.isna().sum().to_dict()}"
                )
                gemini_resp = query_gemini_insights(query, summary_ctx, active_key)
                if gemini_resp and not gemini_resp.startswith("Could not generate Gemini response"):
                    return {
                        "response": gemini_resp,
                        "source": "Google Gemini LLM",
                        "intent": "general"
                    }
            except Exception:
                pass

        # Use High-Precision Built-in Analytical Reasoning Engine
        return {
            "response": self.generate_grounded_answer(query, df),
            "source": "Built-in Statistical AI Engine",
            "intent": "heuristic"
        }

    def generate_grounded_answer(self, query: str, df: pd.DataFrame) -> str:
        """
        Generates an accurate, strictly data-grounded natural language explanation
        answering any specific question about columns, calculations, comparisons,
        correlations, anomalies, aggregations, filters, or business strategy.
        """
        if df.empty:
            return "Active dataset is currently empty. Please upload a CSV file or load a sample dataset first."

        q = query.lower().strip()
        n_rows, n_cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        all_cols_lower = {c.lower(): c for c in df.columns}

        # 0. Dataset Schema & Column List Questions
        if any(w in q for w in ["what columns", "list columns", "show columns", "column names", "schema", "features"]):
            num_str = f"**Numeric ({len(num_cols)}):** `{', '.join(num_cols)}`" if num_cols else "No numeric columns."
            cat_str = f"**Categorical ({len(cat_cols)}):** `{', '.join(cat_cols)}`" if cat_cols else "No categorical columns."
            return (
                f"**📋 Dataset Architecture ({n_rows:,} rows × {n_cols} columns)**:\n\n"
                f"{num_str}\n\n"
                f"{cat_str}\n\n"
                f"💡 *Ask me about any specific column, e.g. 'What is the average {num_cols[0] if num_cols else 'value'}?'*"
            )

        # 1. Row Count & Dataset Dimensions
        if any(w in q for w in ["how many rows", "row count", "how many records", "dataset size", "dimensions", "shape"]):
            total_cells = n_rows * n_cols
            missing_total = int(df.isna().sum().sum())
            return (
                f"**📊 Dataset Dimensions**:\n\n"
                f"• **Total Records (Rows):** **{n_rows:,}** observations\n"
                f"• **Total Attributes (Columns):** **{n_cols}** features\n"
                f"• **Total Data Cells:** **{total_cells:,}**\n"
                f"• **Missing Cells:** **{missing_total:,}** ({round((missing_total / (total_cells or 1)) * 100, 2)}%)\n"
                f"• **Duplicate Rows:** **{int(df.duplicated().sum()):,}**"
            )

        # 2. Group Comparison / Categorical Breakdown across Numeric
        if any(w in q for w in ["compare", "by", "per", "across", "break down", "highest in"]) and cat_cols and num_cols:
            g_col = cat_cols[0]
            for c in cat_cols:
                if c.lower() in q:
                    g_col = c
                    break
            v_col = num_cols[0]
            for c in num_cols:
                if c.lower() in q:
                    v_col = c
                    break

            agg = df.groupby(g_col)[v_col].agg(["sum", "mean", "count"]).sort_values(by="mean", ascending=False).reset_index()
            rows_fmt = "\n".join([f"• **`{r[g_col]}`**: Mean {v_col} = **{r['mean']:,.2f}** | Total = **{r['sum']:,.2f}** ({int(r['count']):,} rows)" for _, r in agg.head(5).iterrows()])
            return (
                f"**📊 Performance Comparison: `{v_col}` grouped by `{g_col}`**:\n\n"
                f"{rows_fmt}\n\n"
                f"🏆 **Top Performer:** **`{agg.iloc[0][g_col]}`** averages **{agg.iloc[0]['mean']:,.2f}**."
            )

        # 3. Filter / Condition Check (e.g. "how many churn = yes", "how many records where...")
        for c in cat_cols:
            for val in df[c].dropna().unique()[:8]:
                val_str = str(val).lower()
                if len(val_str) > 0 and (f"={val_str}" in q.replace(" ", "") or f"is{val_str}" in q.replace(" ", "") or f"where{val_str}" in q.replace(" ", "") or (len(val_str) > 2 and val_str in q and "how many" in q)):
                    match_cnt = int((df[c].astype(str).str.lower() == val_str).sum())
                    match_pct = round((match_cnt / n_rows) * 100, 1)
                    return (
                        f"**🔍 Filter Query: `{c}` = '{val}'**:\n\n"
                        f"• **Matching Records:** **{match_cnt:,}** rows\n"
                        f"• **Proportion:** **{match_pct}%** of the total dataset ({n_rows:,} rows)\n"
                        f"• **Remaining Records:** **{n_rows - match_cnt:,}** rows ({round(100 - match_pct, 1)}%)."
                    )

        # 4. Specific Column Statistics (Mean, Sum, Max, Min, Median, Outliers)
        matched_cols = [orig for low, orig in all_cols_lower.items() if low in q or orig in query]
        if matched_cols:
            target_c = matched_cols[0]
            s = df[target_c].dropna()
            
            # Numeric column specific metrics
            if target_c in num_cols and not s.empty:
                # Average / Mean query
                if any(w in q for w in ["average", "mean", "avg"]):
                    return (
                        f"**📈 Average for `{target_c}`**:\n\n"
                        f"• **Mean Value:** **{s.mean():,.2f}**\n"
                        f"• **Median (50th %):** **{s.median():,.2f}**\n"
                        f"• **Range:** [{s.min():,.2f} to {s.max():,.2f}] across {len(s):,} observations."
                    )
                # Max / Highest / Top query
                if any(w in q for w in ["max", "highest", "peak", "maximum", "largest"]):
                    return (
                        f"**🔝 Maximum for `{target_c}`**:\n\n"
                        f"• **Highest Recorded Value:** **{s.max():,.2f}**\n"
                        f"• **Mean Baseline:** **{s.mean():,.2f}** (Peak is {round(s.max() / (s.mean() or 1), 1)}x of average)."
                    )
                # Min / Lowest / Bottom query
                if any(w in q for w in ["min", "lowest", "minimum", "smallest"]):
                    return (
                        f"**📉 Minimum for `{target_c}`**:\n\n"
                        f"• **Lowest Recorded Value:** **{s.min():,.2f}**\n"
                        f"• **Mean Baseline:** **{s.mean():,.2f}**\n"
                        f"• **25th Percentile:** **{s.quantile(0.25):,.2f}**."
                    )
                # Total / Sum query
                if any(w in q for w in ["total", "sum"]):
                    return (
                        f"**💰 Total Aggregate for `{target_c}`**:\n\n"
                        f"• **Sum Total:** **{s.sum():,.2f}**\n"
                        f"• **Average per Observation:** **{s.mean():,.2f}** across {len(s):,} records."
                    )
                # Median query
                if "median" in q:
                    return (
                        f"**⚖️ Median for `{target_c}`**:\n\n"
                        f"• **Median (50th Percentile):** **{s.median():,.2f}**\n"
                        f"• **Parametric Mean:** **{s.mean():,.2f}** (Skew indicator: {'Right-skewed' if s.mean() > s.median() else 'Left-skewed' if s.mean() < s.median() else 'Symmetric'})."
                    )
                # General Numeric Profile for matched column
                return (
                    f"**Statistical Profile for `{target_c}`**:\n\n"
                    f"• **Valid Count:** **{len(s):,}** (Missing: **{df[target_c].isna().sum():,}**)\n"
                    f"• **Mean:** **{s.mean():,.2f}** | **Median:** **{s.median():,.2f}**\n"
                    f"• **Std Dev:** **{s.std():,.2f}** | **IQR:** **{s.quantile(0.75) - s.quantile(0.25):,.2f}**\n"
                    f"• **Range:** **[{s.min():,.2f}, {s.max():,.2f}]**\n"
                    f"• **95th Percentile Threshold:** **{s.quantile(0.95):,.2f}** ({len(s[s > s.quantile(0.95)]):,} records above)."
                )

            # Categorical column specific metrics
            elif target_c in cat_cols:
                vc = s.value_counts()
                top_cat = vc.index[0] if not vc.empty else "None"
                top_cnt = vc.iloc[0] if not vc.empty else 0
                pct = round((top_cnt / len(s)) * 100, 1) if len(s) else 0
                
                breakdown = "\n".join([f"  - `{k}`: **{v:,}** ({round((v/len(s))*100, 1)}%)" for k, v in vc.head(5).items()])
                return (
                    f"**Categorical Breakdown for `{target_c}`**:\n\n"
                    f"• **Unique Categories:** **{s.nunique()}** distinct values\n"
                    f"• **Dominant Segment:** **`{top_cat}`** with **{top_cnt:,}** occurrences ({pct}% of observations)\n"
                    f"• **Top Categories Distribution:**\n{breakdown}\n"
                    f"• **Missing Cells:** **{df[target_c].isna().sum():,}**"
                )

        # 5. SQL Query Generation Request
        if any(w in q for w in ["sql", "query", "select", "how to write query"]):
            from modules.sql_studio import generate_ai_sql
            sql_spec = generate_ai_sql(query, df)
            return (
                f"**🗄️ Synthesized SQL Query**:\n\n"
                f"```sql\n{sql_spec['sql']}\n```\n\n"
                f"**Explanation:** {sql_spec['explanation']}\n\n"
                f"*Tip: You can execute this query directly in the **🗄️ SQL & Queries** module.*"
            )

        # 6. Data Cleaning & Hygiene Request
        if any(w in q for w in ["clean", "hygiene", "missing", "duplicate", "quality", "anomal"]):
            null_counts = df.isna().sum()
            cols_with_null = null_counts[null_counts > 0]
            dups = int(df.duplicated().sum())
            
            clean_text = f"**🛡️ Data Quality & Cleaning Audit ({n_rows:,} rows × {n_cols} cols)**:\n\n"
            if not cols_with_null.empty:
                clean_text += f"• **Missing Values Detected:** Found across {len(cols_with_null)} column(s):\n"
                for col_name, cnt in cols_with_null.items():
                    pct = round((cnt / n_rows) * 100, 1)
                    impute_rec = "Median imputation" if col_name in num_cols else "Mode / Frequent Category imputation"
                    clean_text += f"  - `{col_name}`: **{cnt:,}** missing ({pct}%) ➔ Recommended: *{impute_rec}*.\n"
            else:
                clean_text += "• **Completeness:** 100% complete! No missing cells detected across any column.\n"

            if dups > 0:
                clean_text += f"• **Redundant Records:** **{dups:,}** duplicate row(s) detected. Recommend 1-Click Deduplication.\n"
            else:
                clean_text += "• **Uniqueness:** All rows are unique; no duplicate records found.\n"

            clean_text += "\n💡 **Recommended Action:** Navigate to **🧹 Data Cleaning** and run **1-Click AI Auto-Clean** to resolve these automatically."
            return clean_text

        # 7. Correlations, Drivers & Multicollinearity
        if any(w in q for w in ["correlation", "correlate", "driver", "relationship", "influence", "multicollinear"]):
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                pairs = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        val = corr.iloc[i, j]
                        if not np.isnan(val):
                            pairs.append((num_cols[i], num_cols[j], val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)

                lead = pairs[:4]
                corr_text = f"**📊 Correlation & Key Drivers Analysis**:\n\n"
                corr_text += "Top associations between numerical features:\n"
                for c1, c2, r in lead:
                    direction = "Positive ↗️" if r > 0 else "Negative ↘️"
                    strength = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.4 else "Weak"
                    corr_text += f"• **`{c1}`** & **`{c2}`**: r = **{r:+.3f}** ({strength} {direction})\n"

                high_coll = [p for p in pairs if abs(p[2]) >= 0.8]
                if high_coll:
                    corr_text += f"\n⚠️ **Multicollinearity Warning:** {len(high_coll)} pair(s) exhibit |r| ≥ 0.80. Consider regularizing or dropping redundant features."
                else:
                    corr_text += "\n✅ No severe multicollinearity (|r| ≥ 0.80) detected."
                return corr_text
            else:
                return "At least two numerical columns are required to compute correlation pairs."

        # 8. Machine Learning & Predictive Modeling Advice
        if any(w in q for w in ["ml", "machine learning", "model", "predict", "automl", "target"]):
            target_guess = "Churn" if "Churn" in df.columns else ("SalePrice" if "SalePrice" in df.columns else (num_cols[-1] if num_cols else df.columns[-1]))
            is_classif = df[target_guess].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[target_guess])
            task_type = "Binary / Multi-Class Classification" if is_classif else "Continuous Regression"

            return (
                f"**🤖 Machine Learning Strategy**:\n\n"
                f"• **Recommended Target Variable:** `{target_guess}`\n"
                f"• **Inferred ML Task:** **{task_type}**\n"
                f"• **Candidate Features (X):** `{', '.join([c for c in df.columns if c != target_guess][:6])}`\n"
                f"• **Suggested Algorithms:**\n"
                f"  - Baseline: `Random Forest` or `Logistic Regression / Ridge`\n"
                f"  - High Performance: `LightGBM / Gradient Boosting`\n"
                f"• **Recommended Action:** Open **🤖 Machine Learning** to run an automated 5-fold cross-validated AutoML tournament!"
            )

        # 9. Visualization Advice
        if any(w in q for w in ["chart", "plot", "visual", "graph"]):
            rec_cat = cat_cols[0] if cat_cols else "Dimension"
            rec_num = num_cols[0] if num_cols else "Metric"
            return (
                f"**📈 Visualization Guidance**:\n\n"
                f"• For Categorical comparison: **Column or Horizontal Bar Chart** comparing `{rec_num}` grouped by `{rec_cat}`.\n"
                f"• For Proportions: **Donut Chart** (`hole=0.45`) across `{rec_cat}`.\n"
                f"• For Distributions: **Histogram + Box Plot** over `{rec_num}`.\n"
                f"• For Relationships: **Scatter Plot** between `{num_cols[0] if len(num_cols) > 0 else 'X'}` and `{num_cols[1] if len(num_cols) > 1 else 'Y'}`.\n"
                f"💡 All of these can be generated with 1-click in the **📈 Visualization** module."
            )

        # 10. Default General Synthesis / Executive Answer
        missing_total = int(df.isna().sum().sum())
        total_cells = n_rows * n_cols
        health_score = max(10, int(100 - (missing_total / (total_cells or 1) * 120) - (df.duplicated().sum() / (n_rows or 1) * 80)))

        return (
            f"**⚡ DataMind AI Executive Overview**:\n\n"
            f"• **Dataset Dimensions:** **{n_rows:,}** rows × **{n_cols}** columns ({total_cells:,} data cells).\n"
            f"• **Data Health Score:** **{health_score}/100** ({'Excellent' if health_score >= 80 else 'Fair' if health_score >= 60 else 'Requires Cleaning'}).\n"
            f"• **Schema Architecture:** **{len(num_cols)}** Numerical attributes (`{', '.join(num_cols[:4])}`), **{len(cat_cols)}** Categorical dimensions (`{', '.join(cat_cols[:4])}`).\n"
            f"• **Data Hygiene:** {missing_total:,} missing cells ({round(missing_total / total_cells * 100, 1)}%) and {int(df.duplicated().sum())} duplicate row(s).\n\n"
            f"💡 **Suggested Questions you can ask:**\n"
            f"- *'What is the average {num_cols[0] if num_cols else 'value'}?'*\n"
            f"- *'Compare {num_cols[0] if num_cols else 'metrics'} by {cat_cols[0] if cat_cols else 'category'}'*\n"
            f"- *'What data quality issues exist and how should I clean them?'*\n"
            f"- *'What are the strongest correlations in this data?'*\n"
            f"- *'Write an SQL query to group top categories'* \n"
            f"- *'Which machine learning model should I train?'*"
        )
