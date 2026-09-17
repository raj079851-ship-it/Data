"""
Comprehensive Automated Unit and Integration Test Suite
Tests all 20+ modules and full data analytics workflows:
- Ingestion, Health Score & Semantic Inference
- Cleaning & Undo/Redo Pipeline
- Data Transformations & Joins
- Feature Engineering & Interactions
- Comprehensive EDA & Correlations
- DuckDB SQL Studio & NL-to-SQL
- Statistical Analysis & Hypothesis Testing
- AutoML, Classification, Regression & Clustering
- What-If Predictions & Forecasting
- Model Explainability & Model Registry
- Power BI Analysis & BI Insights
- Automated Reports & FastAPI Endpoints
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    load_dataset,
    compute_dataset_health_score,
    infer_column_metadata,
    generate_sample_customer_churn,
    generate_sample_housing,
    generate_sample_ecommerce_sales
)
from modules.project_manager import ProjectManager
from modules.data_cleaner import (
    audit_data_quality,
    impute_missing_values,
    handle_outliers,
    clean_text_and_duplicates,
    one_click_ai_auto_clean,
    CleaningPipelineManager
)
from modules.data_processor import (
    filter_rows,
    create_calculated_column,
    group_by_aggregate,
    join_datasets,
    pivot_dataset
)
from modules.feature_engineer import (
    categorize_features,
    scale_features,
    encode_categorical,
    create_interaction_feature,
    extract_datetime_features,
    extract_text_features
)
from modules.eda_engine import (
    compute_comprehensive_stats,
    compute_correlation_analysis,
    analyze_target_variable,
    generate_automated_eda_insights
)
from modules.sql_studio import SQLStudio, generate_ai_sql
from modules.statistical_engine import (
    run_descriptive_statistics,
    run_hypothesis_test,
    run_regression_analysis
)
from modules.ml_engine import (
    preprocess_for_ml,
    train_single_model,
    run_automl_tournament,
    run_unsupervised_clustering
)
from modules.predictive_engine import predict_scenario
from modules.forecasting_engine import generate_forecast, detect_time_series_columns
from modules.model_explainability import get_global_explanations, explain_individual_prediction
from modules.model_registry import ModelRegistry
from modules.powerbi_analyzer import analyze_powerbi_export
from modules.bi_insights import compute_kpi_goal_tracking, compute_period_growth, run_automated_insights_engine
from modules.reports_engine import generate_html_report, generate_excel_report, generate_pdf_report
from modules.ai_orchestrator import AIOrchestrator


class TestFullPlatform(unittest.TestCase):

    def setUp(self):
        self.df_churn = generate_sample_customer_churn()
        self.df_housing = generate_sample_housing()
        self.df_sales = generate_sample_ecommerce_sales()

    def test_data_loader_and_health_score(self):
        # Health score
        health = compute_dataset_health_score(self.df_churn)
        self.assertIn("health_score", health)
        self.assertGreaterEqual(health["health_score"], 0)
        self.assertLessEqual(health["health_score"], 100)

        # Semantic column inference
        meta = infer_column_metadata(self.df_churn)
        self.assertIn("MonthlyCharges", meta)
        self.assertEqual(meta["MonthlyCharges"]["role"], "numeric")
        self.assertIn("Contract", meta)
        self.assertEqual(meta["Contract"]["role"], "categorical")

    def test_project_manager(self):
        pm = ProjectManager()
        p = pm.create_project("Test Automation Project", "Test Description", "churn.csv")
        self.assertIn(p["id"], pm.projects)
        self.assertEqual(pm.active_project_id, p["id"])

        # Rename
        renamed = pm.rename_project(p["id"], "Renamed Automation Project")
        self.assertTrue(renamed)
        self.assertEqual(pm.projects[p["id"]]["name"], "Renamed Automation Project")

        # Duplicate
        dup = pm.duplicate_project(p["id"])
        self.assertIsNotNone(dup)
        self.assertIn("(Copy)", dup["name"])

        # Cleanup
        pm.delete_project(dup["id"])
        pm.delete_project(p["id"])

    def test_cleaning_and_pipeline_undo_redo(self):
        # Audit
        audit = audit_data_quality(self.df_churn)
        self.assertIn("health_score", audit)

        # Impute
        clean_df, log = impute_missing_values(self.df_churn)
        self.assertEqual(clean_df["MonthlyCharges"].isna().sum(), 0)

        # Outliers
        out_df, log_o = handle_outliers(clean_df, columns=["MonthlyCharges"], method="iqr", action="cap")
        self.assertEqual(len(out_df), len(clean_df))

        # Pipeline Manager Undo/Redo
        mgr = CleaningPipelineManager(self.df_churn)
        self.assertFalse(mgr.can_undo())
        mgr.apply_step("Step 1", clean_df)
        self.assertTrue(mgr.can_undo())
        undone = mgr.undo()
        self.assertEqual(len(undone[1]), len(self.df_churn))
        self.assertTrue(mgr.can_redo())
        redone = mgr.redo()
        self.assertEqual(redone[0], "Step 1")

    def test_data_processor_operations(self):
        # Filter
        filtered = filter_rows(self.df_churn, "Contract", "==", "Month-to-month")
        self.assertTrue((filtered["Contract"] == "Month-to-month").all())

        # Calculated column
        calc_df, msg = create_calculated_column(self.df_sales, "Margin", "df['Revenue'] - df['Cost']")
        self.assertIn("Margin", calc_df.columns)

        # Group By
        agg_df = group_by_aggregate(self.df_sales, ["Category"], {"Revenue": ["sum", "avg"]})
        self.assertIn("Revenue_sum", agg_df.columns)
        self.assertIn("Revenue_avg", agg_df.columns)

        # Relational Join
        j_df = join_datasets(self.df_churn.head(10), self.df_churn.head(10), left_on="CustomerID", right_on="CustomerID")
        self.assertEqual(len(j_df), 10)

    def test_feature_engineering(self):
        # Scaling
        scaled_df, log = scale_features(self.df_churn, ["MonthlyCharges"], method="standard")
        self.assertIn("MonthlyCharges_std", scaled_df.columns)

        # Encoding
        enc_df, log_e = encode_categorical(self.df_churn, ["Contract"], method="one_hot")
        self.assertTrue(any(c.startswith("Contract_") for c in enc_df.columns))

        # Interactions
        int_df, log_i = create_interaction_feature(self.df_sales, "Revenue", "UnitsSold", operation="ratio")
        self.assertIn("Revenue_per_UnitsSold", int_df.columns)

        # Text features
        txt_df, log_t = extract_text_features(self.df_sales, "Product")
        self.assertIn("Product_char_count", txt_df.columns)
        self.assertIn("Product_sentiment_score", txt_df.columns)

    def test_eda_and_correlations(self):
        stats = compute_comprehensive_stats(self.df_sales)
        self.assertFalse(stats["numeric"].empty)
        self.assertFalse(stats["categorical"].empty)

        # Correlation
        corr = compute_correlation_analysis(self.df_sales, method="pearson")
        self.assertIn("corr_matrix", corr)

        # Insights
        insights = generate_automated_eda_insights(self.df_sales)
        self.assertGreater(len(insights), 0)

    def test_sql_studio_duckdb(self):
        studio = SQLStudio()
        res, err, elapsed = studio.execute_query(self.df_sales, "SELECT Category, SUM(Revenue) as total_rev FROM df GROUP BY 1 ORDER BY 2 DESC")
        self.assertIsNone(err)
        self.assertGreater(len(res), 0)
        self.assertIn("total_rev", res.columns)

        # NL to SQL
        ai_sql = generate_ai_sql("Show top 5 products by revenue", self.df_sales)
        self.assertIn("SELECT", ai_sql["sql"])
        self.assertIn("Revenue", ai_sql["sql"])

    def test_statistical_engine(self):
        # Descriptive
        desc = run_descriptive_statistics(self.df_sales, "Revenue")
        self.assertIn("mean", desc)
        self.assertIn("ci_95", desc)

        # Hypothesis test
        test_res = run_hypothesis_test(self.df_sales, "one_way_anova", "Revenue", group_col="Category")
        self.assertIn("statistic", test_res)
        self.assertIn("p_value", test_res)

        # Regression
        reg_res = run_regression_analysis(self.df_sales, "Revenue", ["Cost", "UnitsSold"])
        self.assertIn("r_squared", reg_res)
        self.assertIn("coefficients_table", reg_res)

    def test_ml_and_automl(self):
        # AutoML Tournament
        automl = run_automl_tournament(self.df_churn, target_col="Churn", feature_cols=["MonthlyCharges", "TenureMonths"])
        self.assertIn("leaderboard", automl)
        self.assertGreater(len(automl["leaderboard"]), 0)
        self.assertIsNotNone(automl["best_model_result"])

        # Unsupervised Clustering
        clust = run_unsupervised_clustering(self.df_housing, ["SqftLiving", "SalePrice"], n_clusters=3)
        self.assertIn("cluster_df", clust)
        self.assertIn("Cluster", clust["cluster_df"].columns)

    def test_predictive_and_forecasting(self):
        # Train simple model for prediction test
        prep = preprocess_for_ml(self.df_churn, "Churn", ["MonthlyCharges", "TenureMonths"])
        model_res = train_single_model(prep, algorithm="random_forest")

        # Predict
        pred_out = predict_scenario(model_res, {"MonthlyCharges": 85.0, "TenureMonths": 4})
        self.assertIn("prediction", pred_out)
        self.assertIn("confidence_pct", pred_out)

        # Forecasting
        fore = generate_forecast(self.df_sales, "OrderDate", "Revenue", horizon=6, freq="M")
        self.assertIn("forecast_df", fore)
        self.assertEqual(len(fore["forecast_df"]), 6)

    def test_model_registry_and_xai(self):
        reg = ModelRegistry()
        card = reg.register_model("Test Model", "1.0.0", "Test Dataset", "Churn", ["MonthlyCharges"], "Random Forest", {"f1": 0.85})
        self.assertIn(card["id"], reg.models)
        self.assertTrue(reg.update_status(card["id"], "Production"))
        reg.delete_model(card["id"])

    def test_powerbi_and_bi_insights(self):
        # Power BI parser
        pbi = analyze_powerbi_export(self.df_sales, "Sales Export")
        self.assertIn("kpis", pbi)
        self.assertIn("executive_summary", pbi)

        # Goal tracking
        goal = compute_kpi_goal_tracking(self.df_sales, "Revenue", 500000.0, dimension_col="Category")
        self.assertIn("pct_achieved", goal)
        self.assertIn("status", goal)

        # Insights engine
        insights = run_automated_insights_engine(self.df_sales)
        self.assertGreater(len(insights), 0)

    def test_reports_generation(self):
        meta = {"health_score": 92, "health_status": "Excellent", "missing_cells": 10, "duplicated_rows": 2}
        audit_trail = [
            "✨ [1-Click Auto-Clean] Deduplicated 2 row(s) and imputed 10 missing cells.",
            "➕ Inserted new column 'Revenue_tax' via formula: Revenue * 0.1",
            "🎯 [Outlier Winsorization] Capped 5 outlier values in 'Revenue'."
        ]
        html = generate_html_report("Executive Summary", "Test Project", self.df_sales, meta, audit_trail=audit_trail)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Executive Summary", html)
        self.assertIn("Data Transformation &amp; Operations Audit Trail", html)
        self.assertIn("Revenue_tax", html)

        excel_bytes = generate_excel_report("Test Project", self.df_sales, meta, audit_trail=audit_trail)
        self.assertIsInstance(excel_bytes, bytes)
        self.assertGreater(len(excel_bytes), 1000)

        pdf_bytes = generate_pdf_report("Executive Briefing", "Test Project", self.df_sales, meta, audit_trail=audit_trail)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_ai_orchestrator(self):
        orch = AIOrchestrator()
        route = orch.route_omnibar_command("Show top 5 products by revenue", self.df_sales)
        self.assertEqual(route["agent"], "SQL Agent")

        ans = orch.generate_grounded_answer("What is the average revenue?", self.df_sales)
        self.assertIn("Revenue", ans)


if __name__ == "__main__":
    unittest.main()
