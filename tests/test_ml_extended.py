"""
Unit tests for extended Machine Learning engine:
- 12 Regression algorithms (including XGBoost, LightGBM, CatBoost)
- 10 Classification algorithms (including XGBoost, LightGBM, CatBoost)
- Multilabel classification
- 5 Clustering algorithms (KMeans, MiniBatch, DBSCAN, Hierarchical, GMM)
- 3 Dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Association analysis (Apriori & FP-Growth)
- 4 Business use cases (Customer segmentation, Product grouping, Behavioral, Market basket)
"""

import unittest
import pandas as pd
import numpy as np

import modules.ml_engine as mle


class TestMLExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.df = pd.DataFrame({
            "CustomerID": [f"CUST_{i:03d}" for i in range(80)],
            "Age": np.random.randint(18, 70, size=80),
            "Tenure": np.random.randint(1, 60, size=80),
            "MonthlyCharges": np.random.uniform(20.0, 120.0, size=80),
            "TotalCharges": np.random.uniform(100.0, 6000.0, size=80),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=80),
            "Churn": np.random.choice(["Yes", "No"], size=80, p=[0.3, 0.7]),
            "Tier": np.random.choice(["Bronze", "Silver", "Gold"], size=80),
            "TargetMultilabelA": np.random.choice([0, 1], size=80),
            "TargetMultilabelB": np.random.choice([0, 1], size=80),
            "Item": np.random.choice(["Laptop", "Mouse", "Monitor", "Headphones", "Keyboard"], size=80)
        })

    def test_regression_all_12_algorithms(self):
        prep = mle.preprocess_for_ml(self.df, target_col="TotalCharges", feature_cols=["Age", "Tenure", "MonthlyCharges"])
        algos = [
            "linear_regression", "ridge", "lasso", "elastic_net",
            "decision_tree", "random_forest", "gradient_boosting",
            "xgboost", "lightgbm", "catboost", "svr", "knn"
        ]
        for algo in algos:
            res = mle.train_single_model(prep, algorithm=algo)
            self.assertIn("metrics", res)
            self.assertIn("r2", res["metrics"])
            self.assertIn("rmse", res["metrics"])

    def test_classification_all_10_algorithms(self):
        prep = mle.preprocess_for_ml(self.df, target_col="Churn", feature_cols=["Age", "Tenure", "MonthlyCharges"])
        algos = [
            "logistic_regression", "decision_tree", "random_forest",
            "gradient_boosting", "xgboost", "lightgbm", "catboost",
            "svm", "knn", "naive_bayes"
        ]
        for algo in algos:
            res = mle.train_single_model(prep, algorithm=algo)
            self.assertIn("metrics", res)
            self.assertIn("accuracy", res["metrics"])
            self.assertIn("f1", res["metrics"])

    def test_multilabel_classification(self):
        prep = mle.preprocess_for_ml(self.df, target_col=["TargetMultilabelA", "TargetMultilabelB"], feature_cols=["Age", "Tenure", "MonthlyCharges"])
        res = mle.train_single_model(prep, algorithm="random_forest")
        self.assertEqual(res["classification_type"], "multilabel")
        self.assertIn("accuracy", res["metrics"])
        self.assertIn("f1", res["metrics"])

    def test_clustering_all_5_algorithms(self):
        cols = ["Age", "Tenure", "MonthlyCharges"]
        for algo in ["kmeans", "minibatch_kmeans", "dbscan", "hierarchical", "gmm"]:
            res = mle.run_unsupervised_clustering(self.df, columns=cols, n_clusters=3, algorithm=algo)
            self.assertIn("cluster_df", res)
            self.assertIn("Cluster", res["cluster_df"].columns)
            self.assertIn("PCA_1", res["cluster_df"].columns)

    def test_dimensionality_reduction_all_3(self):
        cols = ["Age", "Tenure", "MonthlyCharges"]
        for m in ["pca", "tsne", "umap"]:
            res = mle.run_dimensionality_reduction(self.df, columns=cols, method=m, n_components=2)
            self.assertIn("reduced_df", res)
            self.assertEqual(res["n_components"], 2)

    def test_association_analysis_both_algorithms(self):
        for algo in ["apriori", "fpgrowth"]:
            res = mle.run_association_analysis(self.df, transaction_col="Contract", item_col="Item", min_support=0.01, min_confidence=0.05, algorithm=algo)
            self.assertIn("rules", res)

    def test_business_use_cases(self):
        c_seg = mle.run_customer_segmentation_use_case(self.df)
        self.assertEqual(c_seg.get("use_case"), "Customer Segmentation")

        p_grp = mle.run_product_grouping_use_case(self.df)
        self.assertEqual(p_grp.get("use_case"), "Product Grouping")

        b_seg = mle.run_behavioral_segmentation_use_case(self.df)
        self.assertEqual(b_seg.get("use_case"), "Behavioral Segmentation")

        mba = mle.run_market_basket_use_case(self.df)
        self.assertIn("rules", mba)


if __name__ == "__main__":
    unittest.main()
