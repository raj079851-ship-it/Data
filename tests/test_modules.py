"""
Automated unit and integration test suite for AutoData AI
Verifies data loading, data cleaning, feature engineering, EDA, and AI insights.
"""

import os
import sys
import io
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    generate_sample_customer_churn,
    generate_sample_housing,
    load_csv_data,
    get_dataset_quick_profile
)
from modules.data_cleaner import (
    audit_data_quality,
    impute_missing_values,
    handle_outliers,
    clean_text_and_duplicates,
    convert_column_types,
    one_click_ai_auto_clean
)
from modules.feature_engineer import (
    categorize_features,
    get_feature_recommendations,
    scale_features,
    transform_numerical,
    encode_categorical,
    extract_datetime_features,
    create_interaction_feature
)
from modules.eda_engine import (
    compute_comprehensive_stats,
    compute_correlation_analysis,
    analyze_target_variable
)
from modules.visualizer import (
    plot_distribution,
    plot_categorical_frequency,
    plot_correlation_heatmap
)
from modules.ai_insights import generate_heuristic_insights


class TestAutoDataModules(unittest.TestCase):

    def setUp(self):
        self.churn_df = generate_sample_customer_churn()
        self.housing_df = generate_sample_housing()

    def test_data_loader(self):
        """Test dataset profiling and CSV roundtrip."""
        profile = get_dataset_quick_profile(self.churn_df)
        self.assertGreater(profile["rows"], 0)
        self.assertGreater(profile["columns"], 0)
        self.assertIn("TotalCharges", profile["column_names"])

        # Test CSV load from string buffer
        csv_buffer = io.StringIO(self.churn_df.to_csv(index=False))
        df_loaded, meta = load_csv_data(csv_buffer)
        self.assertEqual(df_loaded.shape[0], self.churn_df.shape[0])

    def test_data_cleaning_audit_and_impute(self):
        """Test quality audit, missing imputation, outlier capping, and auto-clean."""
        audit = audit_data_quality(self.churn_df)
        self.assertIn("health_score", audit)
        self.assertTrue(0 <= audit["health_score"] <= 100)

        # Test imputation
        impute_conf = {"MonthlyCharges": {"strategy": "median"}}
        cleaned, log = impute_missing_values(self.churn_df, impute_conf)
        self.assertEqual(cleaned["MonthlyCharges"].isna().sum(), 0)
        self.assertTrue(len(log) > 0)

        # Test outlier capping on housing dataset
        capped_df, out_log = handle_outliers(self.housing_df, ["SalePrice"], method="iqr", action="cap")
        self.assertLess(capped_df["SalePrice"].max(), self.housing_df["SalePrice"].max())

        # Test 1-Click Auto-Clean
        auto_cleaned, auto_log = one_click_ai_auto_clean(self.churn_df)
        self.assertEqual(auto_cleaned.duplicated().sum(), 0)
        self.assertEqual(auto_cleaned["MonthlyCharges"].isna().sum(), 0)

    def test_feature_engineering(self):
        """Test scaling, encoding, datetime decomposition, and interactions."""
        cats = categorize_features(self.churn_df)
        self.assertIn("MonthlyCharges", cats["numerical"])
        self.assertIn("SignupDate", cats["datetime"])

        recs = get_feature_recommendations(self.churn_df)
        self.assertTrue(len(recs) > 0)

        # Scaling
        scaled_df, s_log = scale_features(self.churn_df, ["TenureMonths"], method="standard")
        self.assertIn("TenureMonths_std", scaled_df.columns)

        # Encoding
        encoded_df, e_log = encode_categorical(self.churn_df, ["Contract"], method="one_hot")
        self.assertTrue(any("Contract_" in col for col in encoded_df.columns))

        # Datetime
        dt_df, dt_log = extract_datetime_features(self.churn_df, ["SignupDate"])
        self.assertIn("SignupDate_year", dt_df.columns)
        self.assertIn("SignupDate_month", dt_df.columns)

        # Interaction
        int_df, int_log = create_interaction_feature(self.churn_df, "TotalCharges", "TenureMonths", operation="ratio")
        self.assertIn("TotalCharges_per_TenureMonths", int_df.columns)

        # 1-Click AI Feature Engineering
        from modules.feature_engineer import one_click_ai_feature_engineering
        auto_fe_df, auto_fe_log = one_click_ai_feature_engineering(self.churn_df)
        self.assertGreater(auto_fe_df.shape[1], self.churn_df.shape[1])
        self.assertTrue(len(auto_fe_log) > 0)

    def test_eda_and_visualizer(self):
        """Test statistical calculations, correlation analysis, and charts."""
        stats = compute_comprehensive_stats(self.churn_df)
        self.assertFalse(stats["numeric"].empty)
        self.assertFalse(stats["categorical"].empty)

        corr = compute_correlation_analysis(self.churn_df)
        self.assertIn("corr_matrix", corr)

        # Target analysis
        target_res = analyze_target_variable(self.churn_df, "Churn")
        self.assertIn("associations", target_res)

        # Visualizations should produce valid Plotly figures
        fig_dist = plot_distribution(self.churn_df, "TenureMonths")
        self.assertIsNotNone(fig_dist)

        fig_heat = plot_correlation_heatmap(corr["corr_matrix"])
        self.assertIsNotNone(fig_heat)

    def test_ai_insights(self):
        """Test heuristic AI executive reporting."""
        audit = audit_data_quality(self.churn_df)
        stats = compute_comprehensive_stats(self.churn_df)
        corr = compute_correlation_analysis(self.churn_df)
        insights = generate_heuristic_insights(self.churn_df, audit, stats, corr)

        self.assertIn("exec_summary", insights)
        self.assertIn("health_score", insights)
        self.assertTrue(len(insights["quality_signals"]) > 0)
        self.assertTrue(len(insights["action_items"]) > 0)

    def test_dashboard_visualization_tools(self):
        """Test the 6 core dashboard visualization tools: Line, Bar, Column, Donut, Pie, and Map."""
        from modules.visualizer import (
            plot_distribution,
            plot_categorical_frequency,
            plot_correlation_heatmap,
            plot_scatter,
            plot_box_by_group,
            plot_dashboard_line_chart,
            plot_dashboard_horizontal_bar,
            plot_dashboard_column_chart,
            plot_dashboard_donut_chart,
            plot_dashboard_pie_chart,
            plot_dashboard_map_chart
        )
        from modules.sql_studio import (
            SQLStudio,
            generate_ai_sql,
            explain_and_optimize_sql
        )

        agg = self.churn_df.groupby("Contract")["MonthlyCharges"].mean().reset_index()

        # 1. Line Chart
        fig_line = plot_dashboard_line_chart(self.churn_df, "TenureMonths", "MonthlyCharges")
        self.assertIsNotNone(fig_line)

        # 2. Horizontal Bar Chart
        fig_bar = plot_dashboard_horizontal_bar(agg, "Contract", "MonthlyCharges")
        self.assertIsNotNone(fig_bar)

        # 3. Vertical Column Chart
        fig_col = plot_dashboard_column_chart(agg, "Contract", "MonthlyCharges")
        self.assertIsNotNone(fig_col)

        # 4. Donut Chart
        fig_donut = plot_dashboard_donut_chart(agg, "Contract", "MonthlyCharges")
        self.assertIsNotNone(fig_donut)

        # 5. Pie Chart
        fig_pie = plot_dashboard_pie_chart(agg, "Contract", "MonthlyCharges")
        self.assertIsNotNone(fig_pie)

        # 6. Map Chart (Fallback & explicit geo coordinates)
        fig_map_fallback = plot_dashboard_map_chart(self.churn_df, "MonthlyCharges", dim_col="Contract")
        self.assertIsNotNone(fig_map_fallback)

        # Test with lat/lon coordinates
        geo_df = self.housing_df.copy()
        geo_df["latitude"] = 37.7749
        geo_df["longitude"] = -122.4194
        fig_map_geo = plot_dashboard_map_chart(geo_df, "SalePrice", lat_col="latitude", lon_col="longitude")
        self.assertIsNotNone(fig_map_geo)

    def test_sql_studio(self):
        """Test DuckDB SQL execution, templates, AI NL-to-SQL, and optimizer."""
        from modules.sql_studio import SQLStudio, generate_ai_sql, explain_and_optimize_sql

        studio = SQLStudio()

        # 1. Basic query
        res_df, err, elapsed = studio.execute_query(self.churn_df, "SELECT * FROM df LIMIT 5")
        self.assertIsNone(err)
        self.assertIsNotNone(res_df)
        self.assertEqual(len(res_df), 5)

        # 2. Aggregation query
        res_agg, err_agg, _ = studio.execute_query(
            self.churn_df,
            "SELECT Contract, COUNT(*) as cnt, ROUND(AVG(TRY_CAST(MonthlyCharges AS DOUBLE)), 2) as avg_chg FROM df GROUP BY Contract"
        )
        self.assertIsNone(err_agg)
        self.assertTrue(len(res_agg) > 0)
        self.assertIn("avg_chg", res_agg.columns)

        # 3. CTE query
        res_cte, err_cte, _ = studio.execute_query(
            self.churn_df,
            "WITH t AS (SELECT * FROM df WHERE TRY_CAST(MonthlyCharges AS DOUBLE) > 50) SELECT COUNT(*) as total FROM t"
        )
        self.assertIsNone(err_cte)
        self.assertEqual(len(res_cte), 1)

        # 4. AI NL-to-SQL generation
        ai_res = generate_ai_sql("top 5 customers by MonthlyCharges", self.churn_df)
        self.assertIn("sql", ai_res)
        self.assertIn("SELECT", ai_res["sql"].upper())
        self.assertIn("MonthlyCharges", ai_res["sql"])

        # 5. SQL Optimizer
        opt = explain_and_optimize_sql("SELECT * FROM df")
        self.assertIn("optimization_tips", opt)
        self.assertTrue(len(opt["optimization_tips"]) > 0)


if __name__ == "__main__":
    import io
    unittest.main()
