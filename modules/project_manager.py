"""
Project Manager Module
Handles analytics project lifecycle:
- Create, rename, duplicate, delete, archive, search projects
- Track project datasets, versioning, cleaning history, transformation history,
  SQL queries, EDA results, charts, models, predictions, dashboards, reports, and AI conversations
- Project activity audit logs
"""

import os
import json
import time
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional

PROJECTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".projects_store.json")


def _get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_default_project(
    name: str = "Customer Churn Intelligence",
    description: str = "Enterprise churn prediction and customer retention analytics.",
    dataset_name: str = "customer_churn.csv"
) -> Dict[str, Any]:
    """Generates a structured default project template."""
    proj_id = f"proj_{int(time.time())}_{name.lower().replace(' ', '_')[:12]}"
    now = _get_timestamp()
    return {
        "id": proj_id,
        "name": name,
        "description": description,
        "created_at": now,
        "updated_at": now,
        "archived": False,
        "active_dataset": dataset_name,
        "datasets": [dataset_name],
        "data_versions": [
            {"version": 1, "created_at": now, "name": "Raw Ingestion", "rows": 608, "columns": 12}
        ],
        "cleaning_history": [],
        "transformation_history": [],
        "saved_queries": [
            {
                "id": "q1",
                "name": "High Value Churners",
                "sql": "SELECT CustomerID, Contract, MonthlyCharges, TotalCharges FROM df WHERE Churn = 'Yes' AND MonthlyCharges > 80 ORDER BY MonthlyCharges DESC LIMIT 10",
                "created_at": now
            }
        ],
        "eda_results": {},
        "charts": [],
        "ml_models": [],
        "predictions": [],
        "dashboards": [
            {
                "id": "dash_default",
                "title": "Executive Overview Dashboard",
                "theme": "Executive Dark",
                "created_at": now,
                "widgets": ["kpi_records", "kpi_churn_rate", "chart_contract", "chart_charges"]
            }
        ],
        "reports": [],
        "ai_conversations": [
            {"role": "assistant", "content": "Welcome to your Customer Churn Intelligence project! Ask me anything about trends, drivers, or predictive modeling.", "timestamp": now}
        ],
        "activity_log": [
            {"action": "Project Initialized", "timestamp": now, "user": "Analyst"}
        ]
    }


class ProjectManager:
    """Manages multi-project workspace state and persistence."""

    def __init__(self):
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.active_project_id: Optional[str] = None
        self._load_from_disk()

        if not self.projects:
            # Seed with 3 professional default projects
            p1 = create_default_project("Customer Churn Intelligence", "Predictive churn analytics and retention drivers.", "Customer Churn (Demo Dataset)")
            p2 = create_default_project("Real Estate Housing Valuation", "Property appraisal, price prediction, and spatial market dynamics.", "Real Estate Housing (Demo Dataset)")
            p3 = create_default_project("Global E-Commerce Omnichannel", "Revenue forecasting, profitability drivers, and multi-channel attribution.", "E-Commerce Sales (Demo Dataset)")
            self.projects[p1["id"]] = p1
            self.projects[p2["id"]] = p2
            self.projects[p3["id"]] = p3
            self.active_project_id = p1["id"]
            self._save_to_disk()

        if not self.active_project_id and self.projects:
            self.active_project_id = next(iter(self.projects.keys()))

    def _load_from_disk(self):
        try:
            if os.path.exists(PROJECTS_FILE):
                with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.projects = data.get("projects", {})
                    self.active_project_id = data.get("active_project_id")
        except Exception:
            self.projects = {}

    def _save_to_disk(self):
        try:
            with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "projects": self.projects,
                    "active_project_id": self.active_project_id
                }, f, indent=2, default=str)
        except Exception:
            pass

    def get_all_projects(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        projs = list(self.projects.values())
        if not include_archived:
            projs = [p for p in projs if not p.get("archived", False)]
        return sorted(projs, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_active_project(self) -> Dict[str, Any]:
        if not self.active_project_id or self.active_project_id not in self.projects:
            if self.projects:
                self.active_project_id = next(iter(self.projects.keys()))
            else:
                p = create_default_project()
                self.projects[p["id"]] = p
                self.active_project_id = p["id"]
        return self.projects[self.active_project_id]

    def set_active_project(self, project_id: str) -> bool:
        if project_id in self.projects:
            self.active_project_id = project_id
            self._save_to_disk()
            return True
        return False

    def create_project(self, name: str, description: str = "", dataset_name: str = "customer_churn.csv") -> Dict[str, Any]:
        p = create_default_project(name=name, description=description, dataset_name=dataset_name)
        self.projects[p["id"]] = p
        self.active_project_id = p["id"]
        self.log_activity("Created project", f"Project '{name}' created.")
        self._save_to_disk()
        return p

    def rename_project(self, project_id: str, new_name: str, new_desc: Optional[str] = None) -> bool:
        if project_id in self.projects:
            old_name = self.projects[project_id]["name"]
            self.projects[project_id]["name"] = new_name
            if new_desc is not None:
                self.projects[project_id]["description"] = new_desc
            self.projects[project_id]["updated_at"] = _get_timestamp()
            self.log_activity("Renamed project", f"Renamed from '{old_name}' to '{new_name}'.", project_id)
            self._save_to_disk()
            return True
        return False

    def duplicate_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        if project_id in self.projects:
            orig = self.projects[project_id]
            dup = copy.deepcopy(orig)
            new_id = f"proj_{int(time.time())}_copy"
            dup["id"] = new_id
            dup["name"] = f"{orig['name']} (Copy)"
            dup["created_at"] = _get_timestamp()
            dup["updated_at"] = _get_timestamp()
            dup["activity_log"].append({
                "action": "Duplicated",
                "timestamp": _get_timestamp(),
                "user": "Analyst"
            })
            self.projects[new_id] = dup
            self.active_project_id = new_id
            self._save_to_disk()
            return dup
        return None

    def delete_project(self, project_id: str) -> bool:
        if project_id in self.projects:
            del self.projects[project_id]
            if self.active_project_id == project_id:
                self.active_project_id = next(iter(self.projects.keys())) if self.projects else None
            self._save_to_disk()
            return True
        return False

    def archive_project(self, project_id: str, archive_state: bool = True) -> bool:
        if project_id in self.projects:
            self.projects[project_id]["archived"] = archive_state
            self.projects[project_id]["updated_at"] = _get_timestamp()
            action = "Archived" if archive_state else "Unarchived"
            self.log_activity(f"{action} project", f"Project {action.lower()}.", project_id)
            self._save_to_disk()
            return True
        return False

    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for p in self.projects.values():
            if q in p["name"].lower() or q in p.get("description", "").lower() or q in p.get("active_dataset", "").lower():
                results.append(p)
        return results

    def log_activity(self, action: str, details: str = "", project_id: Optional[str] = None):
        target_id = project_id or self.active_project_id
        if target_id and target_id in self.projects:
            entry = {
                "action": action,
                "details": details,
                "timestamp": _get_timestamp(),
                "user": "Analyst"
            }
            p = self.projects[target_id]
            if "activity_log" not in p:
                p["activity_log"] = []
            p["activity_log"].insert(0, entry)
            # Keep max 50 log items
            p["activity_log"] = p["activity_log"][:50]
            p["updated_at"] = _get_timestamp()
            self._save_to_disk()
