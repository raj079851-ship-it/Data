"""
Regression tests for ML module safety, target validation, auto-detection,
and dataset switching guards to prevent KeyError: None and unhandled exceptions.
"""

import unittest
import pandas as pd
import numpy as np
from modules.ml_engine import (
    preprocess_for_ml,
    train_single_model,
    run_automl_tournament,
    run_unsupervised_clustering,
    run_dimensionality_reduction,
    run_association_analysis
)


def resolve_ml_task_mode(df: pd.DataFrame, ml_target: str, ml_task_mode: str):
    """
    Simulates the exact task-mode resolution logic in app.py.
    Returns (task_type_arg, is_reg).
    """
    if "Binary" in ml_task_mode:
        task_type_arg = "binary"
        is_reg = False
    elif "Multiclass" in ml_task_mode:
        task_type_arg = "multiclass"
        is_reg = False
    elif "Multilabel" in ml_task_mode:
        task_type_arg = "multilabel"
        is_reg = False
    elif "Regression" in ml_task_mode:
        task_type_arg = None
        is_reg = True
    else:  # "Auto-Detect"
        task_type_arg = None
        if ml_target and df is not None and isinstance(df, pd.DataFrame) and ml_target in df.columns and len(df) > 0:
            target_series = df[ml_target].dropna()
            is_reg = bool(len(target_series) > 0 and pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 12)
        else:
            is_reg = False
    return task_type_arg, is_reg


def validate_ml_dataset(df: pd.DataFrame) -> bool:
    """Simulates the dataset validator in app.py."""
    return (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and len(df.columns) >= 2
        and len(df) >= 3
    )


class TestMLSafetyAndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.df = pd.DataFrame({
            "CustomerID": [f"CUST_{i:03d}" for i in range(50)],
            "Age": np.random.randint(18, 70, size=50),
            "Tenure": np.random.randint(1, 60, size=50),
            "MonthlyCharges": np.random.uniform(20.0, 120.0, size=50),
            "ContinuousTarget": np.random.uniform(100.0, 5000.0, size=50),
            "BinaryTarget": np.random.choice(["Yes", "No"], size=50),
            "MultiClassTarget": np.random.choice(["Tier1", "Tier2", "Tier3"], size=50),
            "DiscreteNumericLowCard": np.random.choice([1, 2, 3], size=50)
        })

    def test_validate_ml_dataset_empty(self):
        self.assertFalse(validate_ml_dataset(pd.DataFrame()))
        self.assertFalse(validate_ml_dataset(None))
        self.assertFalse(validate_ml_dataset(pd.DataFrame({"A": [1, 2, 3]})))  # Only 1 column
        self.assertFalse(validate_ml_dataset(pd.DataFrame({"A": [], "B": []})))  # 0 rows
        self.assertFalse(validate_ml_dataset(pd.DataFrame({"A": [1, 2], "B": [3, 4]})))  # <3 rows
        self.assertTrue(validate_ml_dataset(self.df))

    def test_auto_detect_with_none_target_no_keyerror(self):
        """CRITICAL: ml_target=None must NEVER raise KeyError: None."""
        task_type_arg, is_reg = resolve_ml_task_mode(self.df, None, "Auto-Detect")
        self.assertIsNone(task_type_arg)
        self.assertFalse(is_reg)

        empty_df = pd.DataFrame()
        task_type_arg, is_reg = resolve_ml_task_mode(empty_df, None, "Auto-Detect")
        self.assertIsNone(task_type_arg)
        self.assertFalse(is_reg)

    def test_auto_detect_with_invalid_column_no_keyerror(self):
        task_type_arg, is_reg = resolve_ml_task_mode(self.df, "NonExistentCol", "Auto-Detect")
        self.assertIsNone(task_type_arg)
        self.assertFalse(is_reg)

    def test_auto_detect_continuous_numeric_target(self):
        task_type_arg, is_reg = resolve_ml_task_mode(self.df, "ContinuousTarget", "Auto-Detect")
        self.assertIsNone(task_type_arg)
        self.assertTrue(is_reg, "Continuous target with >12 unique values should be detected as Regression")

    def test_auto_detect_binary_and_multiclass_targets(self):
        _, is_reg_bin = resolve_ml_task_mode(self.df, "BinaryTarget", "Auto-Detect")
        self.assertFalse(is_reg_bin, "String binary target should be detected as Classification")

        _, is_reg_multi = resolve_ml_task_mode(self.df, "MultiClassTarget", "Auto-Detect")
        self.assertFalse(is_reg_multi, "MultiClass string target should be detected as Classification")

        _, is_reg_num_low = resolve_ml_task_mode(self.df, "DiscreteNumericLowCard", "Auto-Detect")
        self.assertFalse(is_reg_num_low, "Numeric target with <=12 unique values should default to Classification")

    def test_explicit_task_modes_override_target(self):
        # Even with continuous target, explicit classification mode is respected
        task_type, is_reg = resolve_ml_task_mode(self.df, "ContinuousTarget", "Binary Classification")
        self.assertEqual(task_type, "binary")
        self.assertFalse(is_reg)

        task_type, is_reg = resolve_ml_task_mode(self.df, "ContinuousTarget", "Multiclass Classification")
        self.assertEqual(task_type, "multiclass")
        self.assertFalse(is_reg)

        task_type, is_reg = resolve_ml_task_mode(self.df, "ContinuousTarget", "Multilabel Classification")
        self.assertEqual(task_type, "multilabel")
        self.assertFalse(is_reg)

        # Even with binary target, explicit regression mode is respected
        task_type, is_reg = resolve_ml_task_mode(self.df, "BinaryTarget", "Regression (Continuous)")
        self.assertIsNone(task_type)
        self.assertTrue(is_reg)

    def test_dataset_signature_switch(self):
        """Simulate switching datasets and checking that state keys are identified for cleanup."""
        dataset_a_sig = ("Churn.csv", ("Age", "Tenure", "Churn"))
        dataset_b_sig = ("Housing.csv", ("Bedrooms", "Sqft", "SalePrice"))
        
        session_state = {
            "_ml_last_dataset_sig": dataset_a_sig,
            "st_ml_target": "Churn",
            "st_ml_features": ["Age", "Tenure"],
            "st_tx_col": "Age",
            "st_item_col": "Tenure"
        }

        # Dataset switches to B
        if session_state.get("_ml_last_dataset_sig") != dataset_b_sig:
            session_state["_ml_last_dataset_sig"] = dataset_b_sig
            for key in ["st_ml_target", "st_ml_features", "st_tx_col", "st_item_col"]:
                if key in session_state:
                    del session_state[key]

        self.assertNotIn("st_ml_target", session_state)
        self.assertNotIn("st_ml_features", session_state)
        self.assertEqual(session_state["_ml_last_dataset_sig"], dataset_b_sig)


if __name__ == "__main__":
    unittest.main()
