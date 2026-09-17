# -*- coding: utf-8 -*-
"""
Unit and regression test suite for Model AutoML automated pipeline:
- Feature Type Detection
- Data Cleaning & Constant/ID Removal
- Missing Value Imputation
- Categorical Encoding
- Target Leakage Detection & Quarantine
- Train/Test Data Split
- Feature Generation
- Multi-Model Benchmarking & Hyperparameter Tuning
- Model Evaluation & Champion Selection
- Model Explainability & Driver Attribution
- Model Persistence & Leaderboard Schema Verification
"""

import unittest
import pandas as pd
import numpy as np

from modules.data_loader import generate_sample_customer_churn, generate_sample_housing
from modules.automl_pipeline import AutoMLPipeline, run_automl_pipeline
from modules.model_registry import ModelRegistry


class TestAutoMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.df_churn = generate_sample_customer_churn()
        cls.df_housing = generate_sample_housing()

    def test_feature_type_detection(self):
        pipeline = AutoMLPipeline()
        types = pipeline.detect_feature_types(self.df_churn, target_col="Churn")
        self.assertIn("numeric_continuous", types)
        self.assertIn("numeric_discrete", types)
        self.assertIn("categorical_low", types)
        self.assertIn("categorical_high", types)
        self.assertIn("MonthlyCharges", types["numeric_continuous"] + types["numeric_discrete"])
        self.assertIn("Contract", types["categorical_low"] + types["categorical_high"])

    def test_data_cleaning(self):
        pipeline = AutoMLPipeline()
        df_test = self.df_churn.copy()
        df_test["constant_col"] = 999
        df_test["id_col"] = [f"id_{i}" for i in range(len(df_test))]

        types = pipeline.detect_feature_types(df_test, target_col="Churn")
        cleaned = pipeline.clean_data(df_test, target_col="Churn", types=types)
        self.assertNotIn("constant_col", cleaned.columns)
        self.assertNotIn("id_col", cleaned.columns)
        self.assertIn("Churn", cleaned.columns)

    def test_missing_value_handling(self):
        pipeline = AutoMLPipeline()
        df_test = self.df_churn.copy()
        df_test.loc[0:10, "MonthlyCharges"] = np.nan
        df_test.loc[0:10, "Contract"] = np.nan

        imputed = pipeline.handle_missing_values(df_test, target_col="Churn")
        self.assertEqual(imputed["MonthlyCharges"].isna().sum(), 0)
        self.assertEqual(imputed["Contract"].isna().sum(), 0)

    def test_categorical_encoding(self):
        pipeline = AutoMLPipeline()
        df_test = pd.DataFrame({
            "cat_low": ["A", "B", "A", "C", "B", "A"] * 5,
            "cat_high": [f"val_{i}" for i in range(30)],
            "num": np.arange(30, dtype=float),
            "target": [0, 1] * 15
        })
        encoded = pipeline.encode_categories(df_test, target_col="target")
        self.assertNotIn("cat_low", encoded.columns)
        self.assertTrue(any(c.startswith("cat_low_") for c in encoded.columns))

    def test_leakage_detection(self):
        pipeline = AutoMLPipeline()
        df_test = self.df_churn.copy()
        # Inject an artificial perfect leak
        df_test["leaky_target_copy"] = (df_test["Churn"] == "Yes").astype(float)

        cleaned_df, leaks = pipeline.detect_leakage(df_test, target_col="Churn", task_type="classification")
        self.assertIn("leaky_target_copy", leaks)
        self.assertNotIn("leaky_target_copy", cleaned_df.columns)

    def test_feature_generation(self):
        pipeline = AutoMLPipeline()
        df_test = pd.DataFrame({
            "num1": np.random.exponential(scale=5.0, size=100),
            "num2": np.random.uniform(10, 50, size=100),
            "date_col": pd.date_range("2024-01-01", periods=100, freq="D"),
            "target": np.random.choice([0, 1], size=100)
        })
        featured = pipeline.generate_features(df_test, target_col="target")
        self.assertTrue(any("year" in c or "month" in c or "day" in c for c in featured.columns))
        self.assertTrue(any("div" in c or "log1p" in c for c in featured.columns))

    def test_classification_automl_pipeline(self):
        res = run_automl_pipeline(
            self.df_churn,
            target_col="Churn",
            prediction_type="classification",
            dataset_name="Customer Churn"
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["task_type"], "classification")
        self.assertEqual(res["target_col"], "Churn")
        self.assertIsNotNone(res["champion_model"])
        self.assertGreater(len(res["leaderboard"]), 5)

        # Verify leaderboard columns
        lb = res["leaderboard"]
        expected_cols = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Training Time (s)"]
        for col in expected_cols:
            self.assertIn(col, lb.columns, f"Expected {col} in classification leaderboard")

        # Verify champion is rank 1
        self.assertIn("🥇 Champion", lb.iloc[0]["Rank"])
        # Verify explanation
        self.assertIn("feature_importances", res["explanation"])
        self.assertIn("narrative", res["explanation"])
        self.assertTrue(len(res["explanation"]["feature_importances"]) > 0)

        # Verify model saved to registry
        self.assertIn("registered_model_card", res)
        self.assertIn("id", res["registered_model_card"])

    def test_regression_automl_pipeline(self):
        res = run_automl_pipeline(
            self.df_housing,
            target_col="SalePrice",
            prediction_type="regression",
            dataset_name="Real Estate Housing"
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["task_type"], "regression")
        self.assertEqual(res["target_col"], "SalePrice")
        self.assertIsNotNone(res["champion_model"])
        self.assertGreater(len(res["leaderboard"]), 5)

        # Verify leaderboard columns
        lb = res["leaderboard"]
        expected_cols = ["Rank", "Model", "MAE", "MSE", "RMSE", "R²", "MAPE", "Training Time (s)"]
        for col in expected_cols:
            self.assertIn(col, lb.columns, f"Expected {col} in regression leaderboard")

        # Verify champion is rank 1
        self.assertIn("🥇 Champion", lb.iloc[0]["Rank"])
        self.assertIn("feature_importances", res["explanation"])
        self.assertIn("narrative", res["explanation"])


if __name__ == "__main__":
    unittest.main()
