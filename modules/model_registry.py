"""
Model Registry Module
Enterprise Machine Learning Model Lifecycle Management:
- Stores Model Cards: Name, Version, Dataset, Features, Target, Algorithm, Hyperparameters, Metrics, Training Date, Status, Author
- Lifecycle Statuses: Development, Validated, Production, Archived
- Model Comparison matrix & Version tracking
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

REGISTRY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".model_registry.json")


def _get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ModelRegistry:
    """Manages registered ML models and lifecycle states."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_from_disk()

        if not self.models:
            # Seed with 2 production-grade default model cards
            m1 = {
                "id": "mod_churn_rf_v1",
                "name": "Customer Churn Classifier",
                "version": "1.2.0",
                "dataset": "Customer Churn (Demo Dataset)",
                "target": "Churn",
                "features": ["Contract", "MonthlyCharges", "TenureMonths", "PaperlessBilling"],
                "algorithm": "Random Forest Classifier",
                "hyperparameters": {"n_estimators": 100, "max_depth": 8, "random_state": 42},
                "metrics": {"accuracy": 0.842, "f1": 0.812, "precision": 0.825, "recall": 0.801, "roc_auc": 0.891},
                "training_date": "2024-03-12 14:22:00",
                "status": "Production",
                "author": "Lead ML Engineer",
                "notes": "Deployed model for real-time customer churn scoring."
            }
            m2 = {
                "id": "mod_price_gbm_v1",
                "name": "Property Valuation Regressor",
                "version": "2.0.1",
                "dataset": "Real Estate Housing (Demo Dataset)",
                "target": "SalePrice",
                "features": ["SqftLiving", "Bedrooms", "Bathrooms", "YearBuilt", "LotSizeAcres"],
                "algorithm": "Gradient Boosting Regressor",
                "hyperparameters": {"n_estimators": 120, "learning_rate": 0.08},
                "metrics": {"r2": 0.884, "rmse": 32140.5, "mae": 24100.2, "mape": 6.8},
                "training_date": "2024-04-05 09:15:00",
                "status": "Validated",
                "author": "Senior Data Scientist",
                "notes": "Evaluated against 2024 property valuation benchmarks."
            }
            self.models[m1["id"]] = m1
            self.models[m2["id"]] = m2
            self._save_to_disk()

    def _load_from_disk(self):
        try:
            if os.path.exists(REGISTRY_FILE):
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    self.models = json.load(f)
        except Exception:
            self.models = {}

    def _save_to_disk(self):
        try:
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.models, f, indent=2, default=str)
        except Exception:
            pass

    def get_all_models(self) -> List[Dict[str, Any]]:
        return sorted(list(self.models.values()), key=lambda x: x.get("training_date", ""), reverse=True)

    def register_model(
        self,
        name: str,
        version: str,
        dataset: str,
        target: str,
        features: List[str],
        algorithm: str,
        metrics: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
        status: str = "Development",
        author: str = "Analyst",
        notes: str = ""
    ) -> Dict[str, Any]:
        """Registers a new model card in the registry."""
        model_id = f"mod_{int(time.time())}_{algorithm.lower().replace(' ', '_')[:10]}"
        card = {
            "id": model_id,
            "name": name,
            "version": version,
            "dataset": dataset,
            "target": target,
            "features": features,
            "algorithm": algorithm,
            "hyperparameters": hyperparameters or {},
            "metrics": metrics,
            "training_date": _get_timestamp(),
            "status": status,
            "author": author,
            "notes": notes
        }
        self.models[model_id] = card
        self._save_to_disk()
        return card

    def update_status(self, model_id: str, new_status: str) -> bool:
        if model_id in self.models:
            self.models[model_id]["status"] = new_status
            self._save_to_disk()
            return True
        return False

    def delete_model(self, model_id: str) -> bool:
        if model_id in self.models:
            del self.models[model_id]
            self._save_to_disk()
            return True
        return False

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Constructs a side-by-side comparison table of selected models."""
        rows = []
        for mid in model_ids:
            if mid in self.models:
                m = self.models[mid]
                row = {
                    "Model Name": m["name"],
                    "Version": m["version"],
                    "Algorithm": m["algorithm"],
                    "Status": m["status"],
                    "Target": m["target"],
                    "Training Date": m["training_date"]
                }
                for k, v in m.get("metrics", {}).items():
                    row[k.upper()] = v
                rows.append(row)
        return pd.DataFrame(rows)
