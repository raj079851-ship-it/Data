"""
Unit tests for modules/ab_testing.py
Verifies conversion tests, continuous mean tests, confidence intervals, effect sizes, power, and sample size sizing.
"""

import unittest
import pandas as pd
import numpy as np

from modules.ab_testing import (
    run_ab_conversion_test,
    run_ab_mean_test,
    calculate_sample_size,
    generate_sample_ab_dataset
)


class TestABTesting(unittest.TestCase):

    def test_sample_datasets_generation(self):
        conv_df = generate_sample_ab_dataset("conversion")
        self.assertGreater(len(conv_df), 1000)
        self.assertIn("Variant", conv_df.columns)
        self.assertIn("Converted", conv_df.columns)

        mean_df = generate_sample_ab_dataset("mean")
        self.assertGreater(len(mean_df), 1000)
        self.assertIn("Pricing_Model", mean_df.columns)
        self.assertIn("Order_Value", mean_df.columns)

    def test_conversion_ab_test_significant(self):
        # 1000 Control with 10% conv (100 conversions)
        # 1000 Treatment with 16% conv (160 conversions) -> clearly significant
        df = pd.DataFrame({
            "group": ["A"] * 1000 + ["B"] * 1000,
            "converted": [1] * 100 + [0] * 900 + [1] * 160 + [0] * 840
        })
        res = run_ab_conversion_test(df, "group", "converted", "A", "B", confidence_level=0.95)
        self.assertNotIn("error", res)
        self.assertEqual(res["control"]["n"], 1000)
        self.assertEqual(res["treatment"]["n"], 1000)
        self.assertAlmostEqual(res["control"]["rate"], 0.10, places=2)
        self.assertAlmostEqual(res["treatment"]["rate"], 0.16, places=2)
        self.assertAlmostEqual(res["difference"], 0.06, places=2)
        self.assertAlmostEqual(res["relative_lift_pct"], 60.0, places=1)
        self.assertTrue(res["is_significant"])
        self.assertLess(res["p_value"], 0.001)
        self.assertEqual(res["winner"], "Treatment (Variant)")
        self.assertGreater(res["power_analysis"]["observed_power"], 0.80)
        self.assertGreater(res["confidence_interval"]["low"], 0.0)

    def test_conversion_ab_test_inconclusive(self):
        # Equal rates (10% vs 10.2%) in small sample
        df = pd.DataFrame({
            "group": ["A"] * 200 + ["B"] * 200,
            "converted": [1] * 20 + [0] * 180 + [1] * 21 + [0] * 179
        })
        res = run_ab_conversion_test(df, "group", "converted", "A", "B", confidence_level=0.95)
        self.assertNotIn("error", res)
        self.assertFalse(res["is_significant"])
        self.assertEqual(res["winner"], "Inconclusive / No Winner")
        self.assertGreater(res["p_value"], 0.05)

    def test_mean_ab_test_significant(self):
        np.random.seed(123)
        c_vals = np.random.normal(50.0, 10.0, 500)
        t_vals = np.random.normal(55.0, 10.0, 500)
        df = pd.DataFrame({
            "variant": ["control"] * 500 + ["treatment"] * 500,
            "revenue": list(c_vals) + list(t_vals)
        })
        res = run_ab_mean_test(df, "variant", "revenue", "control", "treatment", confidence_level=0.95)
        self.assertNotIn("error", res)
        self.assertAlmostEqual(res["difference"], 5.0, delta=1.5)
        self.assertTrue(res["is_significant"])
        self.assertLess(res["p_value"], 0.001)
        self.assertEqual(res["winner"], "Treatment (Variant)")
        self.assertIn("Cohen's d", res["effect_size"]["metric"])
        self.assertGreater(res["effect_size"]["value"], 0.3)

    def test_sample_size_calculator(self):
        # 10% baseline, 20% MDE -> required sample size
        calc_conv = calculate_sample_size(0.10, 20.0, metric_type="conversion", alpha=0.05, power=0.80)
        self.assertNotIn("error", calc_conv)
        self.assertGreater(calc_conv["n_per_variant"], 1000)
        self.assertEqual(calc_conv["total_sample_size"], calc_conv["n_per_variant"] * 2)

        # Continuous metric
        calc_mean = calculate_sample_size(100.0, 10.0, metric_type="mean", alpha=0.05, power=0.80, sd=25.0)
        self.assertNotIn("error", calc_mean)
        self.assertGreater(calc_mean["n_per_variant"], 50)


if __name__ == "__main__":
    unittest.main()
