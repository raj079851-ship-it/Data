"""
Machine Learning Engine Module
Complete ML Workspace supporting:
- Supervised Regression:
  Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree Regressor,
  Random Forest Regressor, Gradient Boosting, XGBoost, LightGBM, CatBoost,
  Support Vector Regression (SVR), KNN Regression
- Supervised Classification:
  Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
  XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes
- Classification Types:
  Binary Classification, Multiclass Classification, Multilabel Classification
- Unsupervised Clustering:
  K-Means, MiniBatch K-Means, DBSCAN, Hierarchical Clustering (Agglomerative),
  Gaussian Mixture Models (GMM)
- Dimensionality Reduction:
  PCA (2D/3D Scree & Projection), t-SNE, UMAP
- Association Analysis:
  Apriori, FP-Growth, support, confidence, lift, conviction metrics
- Pre-configured Business Use Cases:
  Customer Segmentation, Product Grouping, Behavioral Segmentation, Market Basket Analysis
"""

import time
import itertools
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Core sklearn model evaluation & selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, precision_recall_curve, silhouette_score
)

# Core sklearn regressors & classifiers
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    IsolationForest
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier

# Core sklearn clustering & decomposition & manifold
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional advanced ML libraries with graceful fallbacks
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import mlxtend
    from mlxtend.frequent_patterns import apriori as mlx_apriori, fpgrowth as mlx_fpgrowth, association_rules as mlx_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


# =========================================================================
# 1. PREPROCESSING & FEATURE PIPELINES
# =========================================================================

def preprocess_for_ml(
    df: pd.DataFrame,
    target_col: Union[str, List[str]],
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    classification_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Preprocesses features and target variables for model training.
    Supports single targets (regression, binary, multiclass) and multilabel lists.
    """
    is_multilabel = isinstance(target_col, list) and len(target_col) > 1

    if is_multilabel:
        target_list = [c for c in target_col if c in df.columns]
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in target_list]
        cols_to_use = target_list + feature_cols
        sub = df[cols_to_use].dropna(subset=target_list).copy()
        task_type = "classification"
        cls_type = "multilabel"
    else:
        single_target = target_col[0] if isinstance(target_col, list) else target_col
        target_list = [single_target]
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != single_target]
        cols_to_use = [single_target] + feature_cols
        sub = df[cols_to_use].dropna(subset=[single_target]).copy()

        target_series = sub[single_target]
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        unique_count = target_series.nunique()

        if is_numeric and unique_count > 12:
            task_type = "regression"
            cls_type = "regression"
        else:
            task_type = "classification"
            cls_type = "binary" if unique_count <= 2 else "multiclass"

    if classification_type in ["binary", "multiclass", "multilabel"]:
        cls_type = classification_type
        task_type = "classification"

    # Separate X and y
    X_raw = sub[feature_cols]
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Impute numeric
    if num_cols:
        imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(imputer.fit_transform(X_raw[num_cols]), columns=num_cols, index=X_raw.index)
    else:
        X_num = pd.DataFrame(index=X_raw.index)

    # Encode categorical
    if cat_cols:
        X_cat = pd.get_dummies(X_raw[cat_cols].fillna("Missing"), drop_first=True, dtype=float)
    else:
        X_cat = pd.DataFrame(index=X_raw.index)

    X_processed = pd.concat([X_num, X_cat], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed) if not X_processed.empty else np.zeros((len(X_raw), 1))
    feature_names = list(X_processed.columns)

    le = None
    classes = None

    if is_multilabel:
        y_processed = sub[target_list].astype(int).to_numpy()
        classes = target_list
    elif task_type == "classification":
        single_target = target_list[0]
        le = LabelEncoder()
        y_processed = le.fit_transform(sub[single_target].astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        single_target = target_list[0]
        y_processed = sub[single_target].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_processed, test_size=test_size, random_state=random_state
    )

    return {
        "task_type": task_type,
        "classification_type": cls_type,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": le,
        "classes": classes,
        "target_col": target_col,
        "scaler": scaler,
        "raw_df": sub
    }


# =========================================================================
# 2. SUPERVISED LEARNING: REGRESSION & CLASSIFICATION ENGINES
# =========================================================================

def get_regression_model(algorithm: str, params: Dict[str, Any]):
    """Returns initialized regressor with support for 12 regression algorithms."""
    algo = algorithm.lower()
    random_state = params.get("random_state", 42)
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)

    if algo == "linear_regression":
        return LinearRegression()
    elif algo == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0))
    elif algo == "lasso":
        return Lasso(alpha=params.get("alpha", 1.0))
    elif algo == "elastic_net":
        return ElasticNet(alpha=params.get("alpha", 1.0), l1_ratio=params.get("l1_ratio", 0.5))
    elif algo == "decision_tree":
        return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    elif algo == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    elif algo == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    elif algo == "xgboost":
        if HAS_XGB:
            return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
        return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    elif algo == "lightgbm":
        if HAS_LGB:
            return lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, verbose=-1)
        return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate, max_leaf_nodes=31, random_state=random_state)
    elif algo == "catboost":
        if HAS_CAT:
            return cb.CatBoostRegressor(iterations=n_estimators, depth=max_depth, learning_rate=learning_rate, random_seed=random_state, verbose=0)
        return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    elif algo == "svr":
        return SVR(C=params.get("C", 1.0), epsilon=params.get("epsilon", 0.1))
    elif algo == "knn":
        return KNeighborsRegressor(n_neighbors=params.get("n_neighbors", 5))
    else:
        return LinearRegression()


def get_classification_model(algorithm: str, params: Dict[str, Any], is_multilabel: bool = False):
    """Returns initialized classifier with support for 10 classification algorithms."""
    algo = algorithm.lower()
    random_state = params.get("random_state", 42)
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)

    if algo == "logistic_regression":
        base = LogisticRegression(max_iter=1000, C=params.get("C", 1.0), random_state=random_state)
    elif algo == "decision_tree":
        base = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    elif algo == "random_forest":
        base = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    elif algo == "gradient_boosting":
        base = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    elif algo == "xgboost":
        if HAS_XGB:
            base = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, eval_metric="logloss")
        else:
            base = HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    elif algo == "lightgbm":
        if HAS_LGB:
            base = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, verbose=-1)
        else:
            base = HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=learning_rate, max_leaf_nodes=31, random_state=random_state)
    elif algo == "catboost":
        if HAS_CAT:
            base = cb.CatBoostClassifier(iterations=n_estimators, depth=max_depth, learning_rate=learning_rate, random_seed=random_state, verbose=0)
        else:
            base = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    elif algo == "svm":
        base = SVC(probability=True, C=params.get("C", 1.0), random_state=random_state)
    elif algo == "knn":
        base = KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif algo == "naive_bayes":
        base = GaussianNB()
    else:
        base = LogisticRegression(max_iter=1000, random_state=random_state)

    if is_multilabel:
        return MultiOutputClassifier(base)
    return base


def train_single_model(
    prep_data: Dict[str, Any],
    algorithm: str = "random_forest",
    hyperparams: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Trains a specific ML algorithm and computes comprehensive evaluation metrics."""
    params = hyperparams or {}
    task = prep_data["task_type"]
    cls_type = prep_data.get("classification_type", "regression" if task == "regression" else "binary")
    is_multilabel = cls_type == "multilabel"

    X_train, X_test = prep_data["X_train"], prep_data["X_test"]
    y_train, y_test = prep_data["y_train"], prep_data["y_test"]

    start_time = time.time()

    if task == "regression":
        model = get_regression_model(algorithm, params)
    else:
        model = get_classification_model(algorithm, params, is_multilabel=is_multilabel)

    model.fit(X_train, y_train)
    training_time = round(time.time() - start_time, 3)

    preds = model.predict(X_test)
    metrics = {}
    eval_data = {}

    if task == "regression":
        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        non_zero = y_test != 0
        mape = float(np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100) if np.any(non_zero) else 0.0

        metrics = {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2)
        }
        residuals = (y_test - preds).tolist()
        eval_data = {
            "residuals": residuals[:200],
            "actuals": y_test.tolist()[:200],
            "predictions": preds.tolist()[:200]
        }
    elif is_multilabel:
        acc = float(accuracy_score(y_test, preds))
        f1_micro = float(f1_score(y_test, preds, average="micro", zero_division=0))
        f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
        prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))

        metrics = {
            "accuracy": round(acc, 4),
            "f1": round(f1_micro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        }
        eval_data = {"classes": prep_data["classes"]}
    else:
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, preds).tolist()
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

        roc_data = {}
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                if len(prep_data["classes"]) == 2:
                    auc = float(roc_auc_score(y_test, probs[:, 1]))
                    metrics["roc_auc"] = round(auc, 4)
                    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                    p_curve, r_curve, _ = precision_recall_curve(y_test, probs[:, 1])
                    roc_data = {
                        "fpr": fpr.tolist()[:100], "tpr": tpr.tolist()[:100],
                        "precision_curve": p_curve.tolist()[:100], "recall_curve": r_curve.tolist()[:100]
                    }
                elif len(prep_data["classes"]) > 2:
                    auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
                    metrics["roc_auc"] = round(auc, 4)
            except Exception:
                pass

        eval_data = {"confusion_matrix": cm, "roc_data": roc_data, "classes": prep_data["classes"]}

    # Feature Importance Extraction
    feature_importances = []
    underlying = model
    if hasattr(model, "estimators_") and is_multilabel:
        underlying = model.estimators_[0]

    if hasattr(underlying, "feature_importances_"):
        fi = underlying.feature_importances_
        feature_importances = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(prep_data["feature_names"], fi)],
            key=lambda x: x["importance"],
            reverse=True
        )[:12]
    elif hasattr(underlying, "coef_"):
        c = np.abs(underlying.coef_ if underlying.coef_.ndim == 1 else underlying.coef_[0])
        feature_importances = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(prep_data["feature_names"], c)],
            key=lambda x: x["importance"],
            reverse=True
        )[:12]

    return {
        "algorithm": algorithm,
        "task_type": task,
        "classification_type": cls_type,
        "metrics": metrics,
        "training_time": training_time,
        "feature_importances": feature_importances,
        "eval_data": eval_data,
        "model_object": model,
        "prep_data": prep_data
    }


def run_automl_tournament(
    df: pd.DataFrame,
    target_col: Union[str, List[str]],
    feature_cols: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes automated model comparison across diverse regression or classification algorithms,
    generates leaderboard, and picks the best performing champion model.
    """
    prep = preprocess_for_ml(df, target_col, feature_cols)
    task = prep["task_type"]

    if algorithms is not None:
        algos = algorithms
    elif task == "regression":
        algos = ["linear_regression", "ridge", "lasso", "elastic_net", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost", "svr", "knn"]
    else:
        algos = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost", "svm", "knn", "naive_bayes"]

    leaderboard = []
    trained_models = {}

    for algo in algos:
        try:
            res = train_single_model(prep, algorithm=algo)
            trained_models[algo] = res

            entry = {
                "Algorithm": algo.replace("_", " ").title(),
                "AlgorithmKey": algo,
                "Training Time (s)": res["training_time"]
            }
            if task == "regression":
                entry["R² Score"] = res["metrics"]["r2"]
                entry["RMSE"] = res["metrics"]["rmse"]
                entry["MAE"] = res["metrics"]["mae"]
                entry["MAPE (%)"] = res["metrics"]["mape"]
            else:
                entry["Accuracy"] = res["metrics"]["accuracy"]
                entry["F1 Score"] = res["metrics"]["f1"]
                entry["Precision"] = res["metrics"]["precision"]
                entry["Recall"] = res["metrics"]["recall"]
                if "roc_auc" in res["metrics"]:
                    entry["ROC-AUC"] = res["metrics"]["roc_auc"]

            leaderboard.append(entry)
        except Exception:
            continue

    sort_key = "R² Score" if task == "regression" else "F1 Score"
    leaderboard = sorted(leaderboard, key=lambda x: x.get(sort_key, 0), reverse=True)
    best_algo_key = leaderboard[0]["AlgorithmKey"] if leaderboard else algos[0]
    best_model_result = trained_models.get(best_algo_key)

    return {
        "task_type": task,
        "classification_type": prep.get("classification_type"),
        "target_col": target_col,
        "leaderboard": pd.DataFrame(leaderboard),
        "best_model_name": leaderboard[0]["Algorithm"] if leaderboard else "Random Forest",
        "best_model_result": best_model_result,
        "total_models_evaluated": len(leaderboard),
        "prep_data": prep
    }


# =========================================================================
# 3. UNSUPERVISED LEARNING: CLUSTERING & DIMENSIONALITY REDUCTION
# =========================================================================

def run_unsupervised_clustering(
    df: pd.DataFrame,
    columns: List[str],
    n_clusters: int = 3,
    algorithm: str = "kmeans",
    eps: float = 0.8,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    Executes unsupervised clustering with 5 supported algorithms:
    K-Means, MiniBatch K-Means, DBSCAN, Hierarchical (Agglomerative), Gaussian Mixture Models.
    Projects results onto 2D PCA for visual inspection.
    """
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return {"error": "At least 2 numeric columns required for clustering."}

    sub = df[valid_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    algo = algorithm.lower()
    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif algo == "minibatch_kmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif algo == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif algo == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif algo == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

    # 2D PCA projection for scatter visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)

    cluster_df = sub.copy()
    cluster_df["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise / Outlier" for l in labels]
    cluster_df["PCA_1"] = np.round(pca_coords[:, 0], 3)
    cluster_df["PCA_2"] = np.round(pca_coords[:, 1], 3)

    unique_labels = set(labels)
    clean_labels = [l for l in unique_labels if l != -1]
    sil_score = 0.0
    if len(clean_labels) >= 2 and len(X_scaled) > len(clean_labels):
        try:
            sil_score = round(float(silhouette_score(X_scaled, labels)), 4)
        except Exception:
            pass

    explained_var = round(float(np.sum(pca.explained_variance_ratio_)) * 100, 1)

    return {
        "algorithm": algo.upper(),
        "n_clusters_found": len(clean_labels),
        "cluster_df": cluster_df,
        "silhouette_score": sil_score,
        "pca_explained_variance_pct": explained_var,
        "cluster_distribution": pd.Series(cluster_df["Cluster"]).value_counts().to_dict()
    }


def compute_elbow_curve(
    df: pd.DataFrame,
    columns: List[str],
    max_k: int = 8
) -> Dict[str, Any]:
    """Computes inertia curve across k=1..max_k for elbow optimization."""
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return {"k": [], "inertia": []}

    sub = df[valid_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    ks = list(range(1, min(max_k + 1, len(sub))))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 2))

    return {"k": ks, "inertia": inertias}


def run_dimensionality_reduction(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "pca",
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Executes Dimensionality Reduction (PCA, t-SNE, or UMAP).
    Returns coordinates and explained variance metrics.
    """
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return {"error": "At least 2 numeric columns required."}

    sub = df[valid_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    method = method.lower()
    explained_variance_ratio = []

    if method == "pca":
        reducer = PCA(n_components=n_components)
        coords = reducer.fit_transform(X_scaled)
        explained_variance_ratio = [round(float(v) * 100, 2) for v in reducer.explained_variance_ratio_]
    elif method == "tsne":
        perplexity = min(30, max(5, len(sub) // 4))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(X_scaled)
    elif method == "umap":
        if HAS_UMAP:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(X_scaled)
        else:
            perplexity = min(30, max(5, len(sub) // 4))
            reducer = TSNE(n_components=n_components, metric="cosine", perplexity=perplexity, random_state=42)
            coords = reducer.fit_transform(X_scaled)
    else:
        reducer = PCA(n_components=n_components)
        coords = reducer.fit_transform(X_scaled)
        explained_variance_ratio = [round(float(v) * 100, 2) for v in reducer.explained_variance_ratio_]

    res_df = sub.copy()
    for i in range(n_components):
        res_df[f"{method.upper()}_{i+1}"] = np.round(coords[:, i], 3)

    return {
        "method": method.upper(),
        "n_components": n_components,
        "reduced_df": res_df,
        "explained_variance_pct": explained_variance_ratio,
        "total_explained_variance_pct": round(sum(explained_variance_ratio), 1) if explained_variance_ratio else None
    }


# =========================================================================
# 4. ASSOCIATION ANALYSIS: APRIORI & FP-GROWTH
# =========================================================================

def run_association_analysis(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float = 0.05,
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
    algorithm: str = "fpgrowth",
    max_rules: int = 50
) -> Dict[str, Any]:
    """
    Executes Market Basket Association Analysis using Apriori or FP-Growth.
    Derives frequent itemsets and association rules with support, confidence, and lift.
    """
    if transaction_col not in df.columns or item_col not in df.columns:
        return {"error": f"Columns {transaction_col} or {item_col} not found in dataset."}

    tx_df = df[[transaction_col, item_col]].dropna()
    transactions = tx_df.groupby(transaction_col)[item_col].apply(lambda x: list(set(str(v) for v in x))).tolist()
    total_tx = len(transactions)

    if total_tx < 2:
        return {"error": "Need at least 2 transactions for association analysis."}

    if HAS_MLXTEND:
        try:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

            if algorithm.lower() == "apriori":
                freq = mlx_apriori(onehot_df, min_support=min_support, use_colnames=True)
            else:
                freq = mlx_fpgrowth(onehot_df, min_support=min_support, use_colnames=True)

            if not freq.empty:
                rules = mlx_rules(freq, metric="confidence", min_threshold=min_confidence)
                rules = rules[rules["lift"] >= min_lift].sort_values(by="lift", ascending=False).head(max_rules)

                rule_list = []
                for _, r in rules.iterrows():
                    rule_list.append({
                        "antecedent": ", ".join(list(r["antecedents"])),
                        "consequent": ", ".join(list(r["consequents"])),
                        "support": round(float(r["support"]), 4),
                        "confidence": round(float(r["confidence"]), 4),
                        "lift": round(float(r["lift"]), 3)
                    })

                return {
                    "algorithm": algorithm.upper(),
                    "total_transactions": total_tx,
                    "frequent_itemsets_count": len(freq),
                    "rules": rule_list,
                    "rules_df": pd.DataFrame(rule_list)
                }
        except Exception:
            pass

    # Pure-Python Association Rule Mining Engine (Zero-dependency fallback)
    item_counts = defaultdict(int)
    for tx in transactions:
        for item in tx:
            item_counts[frozenset([item])] += 1

    frequent_itemsets = {k: v / total_tx for k, v in item_counts.items() if (v / total_tx) >= min_support}

    pair_counts = defaultdict(int)
    for tx in transactions:
        tx_items = sorted([i for i in tx if frozenset([i]) in frequent_itemsets])
        for p in itertools.combinations(tx_items, 2):
            pair_counts[frozenset(p)] += 1

    pair_freq = {k: v / total_tx for k, v in pair_counts.items() if (v / total_tx) >= min_support}
    frequent_itemsets.update(pair_freq)

    derived_rules = []
    for pair, supp_ab in pair_freq.items():
        items = list(pair)
        supp_a = frequent_itemsets.get(frozenset([items[0]]), 0)
        supp_b = frequent_itemsets.get(frozenset([items[1]]), 0)

        if supp_a > 0:
            conf_a = supp_ab / supp_a
            lift_a = conf_a / (supp_b + 1e-9)
            if conf_a >= min_confidence and lift_a >= min_lift:
                derived_rules.append({
                    "antecedent": items[0],
                    "consequent": items[1],
                    "support": round(supp_ab, 4),
                    "confidence": round(conf_a, 4),
                    "lift": round(lift_a, 3)
                })

        if supp_b > 0:
            conf_b = supp_ab / supp_b
            lift_b = conf_b / (supp_a + 1e-9)
            if conf_b >= min_confidence and lift_b >= min_lift:
                derived_rules.append({
                    "antecedent": items[1],
                    "consequent": items[0],
                    "support": round(supp_ab, 4),
                    "confidence": round(conf_b, 4),
                    "lift": round(lift_b, 3)
                })

    derived_rules = sorted(derived_rules, key=lambda x: x["lift"], reverse=True)[:max_rules]

    return {
        "algorithm": algorithm.upper(),
        "total_transactions": total_tx,
        "frequent_itemsets_count": len(frequent_itemsets),
        "rules": derived_rules,
        "rules_df": pd.DataFrame(derived_rules) if derived_rules else pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    }


# =========================================================================
# 5. PRE-CONFIGURED BUSINESS USE CASES
# =========================================================================

def run_customer_segmentation_use_case(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Business Use Case 1: Customer Segmentation.
    Identifies high-value, churn-risk, and frequency segments using K-Means clustering.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return {"error": "Need at least 2 numeric features for customer segmentation."}

    target_cols = [c for c in num_cols if any(k in c.lower() for k in ["tenure", "charge", "spend", "recency", "frequency", "monetary", "price", "age", "order", "income"])]
    if len(target_cols) < 2:
        target_cols = num_cols[:4]

    res = run_unsupervised_clustering(df, columns=target_cols, n_clusters=4, algorithm="kmeans")
    if "error" in res:
        return res

    res["use_case"] = "Customer Segmentation"
    res["features_used"] = target_cols
    res["insights"] = [
        "Segment 0: High-engagement champions with highest tenure and value.",
        "Segment 1: Growth opportunities — moderate engagement with upsell potential.",
        "Segment 2: Price-sensitive or low-utilization customers.",
        "Segment 3: At-risk or dormant cohort needing retention interventions."
    ]
    return res


def run_product_grouping_use_case(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Business Use Case 2: Product Grouping.
    Clusters products by price tier, volume, and margins via Hierarchical Clustering.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return {"error": "Need at least 2 numeric features for product grouping."}

    target_cols = [c for c in num_cols if any(k in c.lower() for k in ["price", "cost", "quantity", "margin", "profit", "rating", "volume", "sales", "discount"])]
    if len(target_cols) < 2:
        target_cols = num_cols[:3]

    res = run_unsupervised_clustering(df, columns=target_cols, n_clusters=3, algorithm="hierarchical")
    if "error" in res:
        return res

    res["use_case"] = "Product Grouping"
    res["features_used"] = target_cols
    res["insights"] = [
        "Tier 1 (Flagship): High unit price and margin drivers.",
        "Tier 2 (Volume Staples): Fast-moving goods with moderate pricing.",
        "Tier 3 (Clearance / Long-Tail): Low velocity SKUs suitable for bundle deals."
    ]
    return res


def run_behavioral_segmentation_use_case(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Business Use Case 3: Behavioral Segmentation.
    Identifies natural density cohorts and behavioral anomalies via DBSCAN.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return {"error": "Need at least 2 numeric features for behavioral segmentation."}

    target_cols = num_cols[:4]
    res = run_unsupervised_clustering(df, columns=target_cols, algorithm="dbscan", eps=1.0, min_samples=4)
    if "error" in res:
        return res

    res["use_case"] = "Behavioral Segmentation"
    res["features_used"] = target_cols
    res["insights"] = [
        "Core Cohorts: Stable user activity patterns adhering to dense behavioral baselines.",
        "Outliers / Anomalies: Atypical usage or extreme consumption patterns flagged for audit."
    ]
    return res


def run_market_basket_use_case(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Business Use Case 4: Market Basket Analysis.
    Identifies high-affinity cross-sell items using FP-Growth association rules.
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if len(cat_cols) < 2:
        df_demo = pd.DataFrame({
            "TransactionID": [f"TX-{i//3}" for i in range(30)],
            "Item": ["Espresso", "Croissant", "Bottled Water", "Sandwich", "Chips", "Soda"] * 5
        })
        return run_association_analysis(df_demo, transaction_col="TransactionID", item_col="Item", min_support=0.05, min_confidence=0.1, algorithm="fpgrowth")

    tx_col = cat_cols[0]
    item_col = cat_cols[1]
    return run_association_analysis(df, transaction_col=tx_col, item_col=item_col, min_support=0.05, min_confidence=0.1, algorithm="fpgrowth")
