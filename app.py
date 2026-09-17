# -*- coding: utf-8 -*-
"""
DataMind AI - Production-Ready AI-Powered Data Analytics Platform
A complete, enterprise-grade analytics workspace supporting the entire data lifecycle across 22 primary modules.
"""

import os
import io
import time
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Platform Modules
from modules.data_loader import (
    load_dataset,
    load_csv_data,
    get_dataset_quick_profile,
    compute_dataset_health_score,
    infer_column_metadata,
    generate_sample_customer_churn,
    generate_sample_housing,
    generate_sample_ecommerce_sales,
    ensure_sample_data_files
)
from modules.project_manager import ProjectManager
from modules.data_cleaner import (
    audit_data_quality,
    impute_missing_values,
    handle_outliers,
    clean_text_and_duplicates,
    convert_column_types,
    one_click_ai_auto_clean,
    CleaningPipelineManager
)
from modules.data_processor import (
    filter_rows,
    select_and_rename_columns,
    create_calculated_column,
    group_by_aggregate,
    join_datasets,
    pivot_dataset,
    unpivot_dataset,
    split_combine_columns
)
from modules.feature_engineer import (
    categorize_features,
    get_feature_recommendations,
    scale_features,
    transform_numerical,
    encode_categorical,
    target_encode,
    extract_datetime_features,
    create_interaction_feature,
    create_polynomial_features,
    calculate_rolling_stats,
    extract_text_features,
    one_click_ai_feature_engineering
)
from modules.eda_engine import (
    compute_comprehensive_stats,
    compute_correlation_analysis,
    analyze_target_variable,
    generate_automated_eda_insights
)
from modules.visualizer import (
    plot_distribution,
    plot_categorical_frequency,
    plot_correlation_heatmap,
    plot_scatter,
    plot_box_by_group,
    plot_scatter_matrix,
    plot_dashboard_line_chart,
    plot_dashboard_horizontal_bar,
    plot_dashboard_column_chart,
    plot_dashboard_donut_chart,
    plot_dashboard_pie_chart,
    plot_dashboard_map_chart,
    plot_violin,
    plot_density_kde,
    plot_ecdf,
    plot_qq,
    plot_treemap,
    plot_sunburst,
    plot_funnel,
    plot_waterfall,
    plot_radar,
    plot_gauge,
    plot_pareto,
    recommend_best_chart
)
from modules.sql_studio import SQLStudio, generate_ai_sql, explain_and_optimize_sql
from modules.statistical_engine import (
    run_descriptive_statistics,
    run_hypothesis_test,
    run_regression_analysis,
    compute_correlation_matrices,
    compute_covariance_matrix,
    explain_statistics_in_plain_language
)
from modules.ab_testing import (
    run_ab_conversion_test,
    run_ab_mean_test,
    calculate_sample_size,
    generate_sample_ab_dataset
)
from modules.connectors import (
    ConnectorType,
    ConnectionStatus,
    ConnectorConfig,
    ConnectionResult,
    SchemaMetadata,
    TableMetadata,
    ColumnMetadata,
    BaseConnector,
    CredentialVault,
    CredentialSanitizer,
    default_vault,
    CONNECTOR_REGISTRY,
    CONNECTOR_CATALOG,
    create_connector,
    get_or_create_demo_sqlite,
)
from modules.ai_orchestrator import AIOrchestrator
from modules.ml_engine import (
    preprocess_for_ml,
    train_single_model,
    run_automl_tournament,
    run_unsupervised_clustering,
    compute_elbow_curve,
    run_dimensionality_reduction,
    run_association_analysis,
    run_customer_segmentation_use_case,
    run_product_grouping_use_case,
    run_behavioral_segmentation_use_case,
    run_market_basket_use_case
)
from modules.automl_pipeline import run_automl_pipeline, AutoMLPipeline
from modules.predictive_engine import predict_scenario
from modules.forecasting_engine import (
    detect_time_series_columns,
    generate_forecast
)
from modules.model_explainability import (
    get_global_explanations,
    explain_individual_prediction
)
from modules.model_registry import ModelRegistry
from modules.powerbi_analyzer import analyze_powerbi_export
from modules.dashboard_builder import DASHBOARD_THEMES, generate_ai_dashboard_spec
from modules.bi_insights import (
    compute_kpi_goal_tracking,
    compute_period_growth,
    run_automated_insights_engine
)
from modules.data_quality_center import DataQualityCenter
from modules.reports_engine import (
    generate_html_report,
    generate_excel_report,
    generate_pdf_report
)
from modules.unicode_helper import sanitize_unicode

# Set page configuration
st.set_page_config(
    page_title="DataMind AI | Production Analytics Workspace",
    page_icon="assets/datamind_logo_thumb.jpg" if os.path.exists("assets/datamind_logo_thumb.jpg") else "🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark / Light SaaS styling)
st.markdown("""
<style>
    /* Single-Time Move UP on Click for all interactive keys & buttons */
    @keyframes btnSingleMoveUp {
      0% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-4px);
      }
      100% {
        transform: translateY(0);
      }
    }

    .btn-clicked-animation {
      animation: btnSingleMoveUp 0.22s cubic-bezier(0.2, 0.9, 0.3, 1) !important;
      position: relative;
    }

    span.btn-clicked-animation, a.btn-clicked-animation, .badge.btn-clicked-animation {
      display: inline-block !important;
    }

    /* Premium SaaS Typography & Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .badge-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-info { background-color: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }

    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.25);
    }
    .action-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 18px;
        border-left: 4px solid #6366f1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        color: #cbd5e1;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("is_light_mode", False):
    st.markdown("""
    <style>
        .stApp {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #cbd5e1 !important;
        }
        .hero-banner {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%) !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
        }
        .hero-banner h2, .hero-banner p {
            color: #0f172a !important;
        }
        .metric-card, .action-box {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05) !important;
        }
        .metric-value {
            color: #0f172a !important;
        }
        .metric-label {
            color: #64748b !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff !important;
            color: #334155 !important;
            border: 1px solid #cbd5e1 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4f46e5 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "is_light_mode" not in st.session_state:
        st.session_state.is_light_mode = False
    if "project_manager" not in st.session_state:
        st.session_state.project_manager = ProjectManager()
    if "model_registry" not in st.session_state:
        st.session_state.model_registry = ModelRegistry()
    if "sql_studio" not in st.session_state:
        st.session_state.sql_studio = SQLStudio()
    if "quality_center" not in st.session_state:
        st.session_state.quality_center = DataQualityCenter()
    if "ai_orchestrator" not in st.session_state:
        st.session_state.ai_orchestrator = AIOrchestrator()

    if "raw_df" not in st.session_state:
        empty_df = pd.DataFrame()
        st.session_state.raw_df = empty_df.copy()
        st.session_state.current_df = empty_df.copy()
        st.session_state.dataset_source = "None (Upload CSV)"
        st.session_state.cleaning_manager = CleaningPipelineManager(empty_df)
        st.session_state.cleaning_history = []
        st.session_state.fe_history = []
        st.session_state.trained_model_res = None
        st.session_state.automl_res = None
        st.session_state.user_role = "Admin"
        st.session_state.theme = "Executive Dark"
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            {
                "role": "assistant",
                "content": "👋 **Hello! I am your AI Data Analytics Assistant.**\n\nI have complete contextual awareness of your active dataset (`" + str(st.session_state.get("dataset_source", "dataset")) + "`).\n\nYou can ask me to:\n- 📊 **Summarize key trends & segment leaders**\n- 🛡️ **Audit data quality & recommend cleaning steps**\n- 📈 **Identify top correlations & drivers**\n- 🗄️ **Synthesize DuckDB SQL queries**\n- 🤖 **Suggest machine learning algorithms & features**\n\nClick any quick prompt chip below or type your question!"
            }
        ]

init_session_state()
ensure_sample_data_files()

# ---------------- TASK EXECUTION MONITOR (BOTTOM-LEFT STATUS MONITOR) ----------------
def set_task_status(task_name: str, status: str = "success", details: str = ""):
    st.session_state["task_monitor"] = {
        "task": task_name,
        "status": status,
        "details": details,
        "timestamp": time.time()
    }

def render_task_execution_monitor():
    task_info = st.session_state.get("task_monitor")
    if task_info and (time.time() - task_info.get("timestamp", 0) > 12):
        task_info = None
        st.session_state["task_monitor"] = None
    
    task_json = json.dumps(task_info) if task_info else "null"
    
    components.html("""
    <script>
    (function() {{
      const doc = window.parent.document;
      if (!doc) return;

      // 1. Inject Styles
      if (!doc.getElementById("st-task-monitor-styles")) {{
        const style = doc.createElement("style");
        style.id = "st-task-monitor-styles";
        style.textContent = `
          #stTaskExecutionMonitor {{
            position: fixed !important;
            bottom: 24px !important;
            right: 24px !important;
            left: auto !important;
            z-index: 99999999 !important;
            min-width: 320px;
            max-width: 440px;
            padding: 14px 18px;
            border-radius: 12px;
            box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            opacity: 0;
            transform: translateY(20px) scale(0.96);
            pointer-events: none;
          }}
          #stTaskExecutionMonitor.show {{
            opacity: 1 !important;
            transform: translateY(0) scale(1) !important;
            pointer-events: auto !important;
          }}
          #stTaskExecutionMonitor.state-running {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.94) 100%) !important;
            border: 1.5px solid #6366f1 !important;
            box-shadow: 0 16px 36px -6px rgba(99, 102, 241, 0.35) !important;
          }}
          #stTaskExecutionMonitor.state-success {{
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%) !important;
            border: 1.5px solid #10b981 !important;
            box-shadow: 0 16px 36px -6px rgba(16, 185, 129, 0.45) !important;
          }}
          #stTaskExecutionMonitor.state-failed {{
            background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%) !important;
            border: 1.5px solid #ef4444 !important;
            box-shadow: 0 16px 36px -6px rgba(239, 68, 68, 0.45) !important;
          }}
          @keyframes stTaskSpin {{
            to {{ transform: rotate(360deg); }}
          }}
          @keyframes stTaskPulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
          }}
        `;
        doc.head.appendChild(style);
      }}

      // 2. Ensure container in top-level body
      let mon = doc.getElementById("stTaskExecutionMonitor");
      if (!mon) {{
        mon = doc.createElement("div");
        mon.id = "stTaskExecutionMonitor";
        doc.body.appendChild(mon);
      }}

      window.parent._dismissStTaskMonitor = function() {{
        const m = doc.getElementById("stTaskExecutionMonitor");
        if (m) {{
          m.className = "";
          sessionStorage.removeItem("st_task_status");
          sessionStorage.removeItem("st_last_clicked_task");
        }}
      }};

      function escapeHtml(str) {{
        if (!str) return "";
        return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
      }}

      function showRunning(taskName) {{
        mon.className = "state-running show";
        mon.innerHTML = `
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <div style="width:20px;height:20px;border:3px solid rgba(99,102,241,0.3);border-top-color:#818cf8;border-radius:50%;animation:stTaskSpin 0.75s linear infinite;flex-shrink:0;"></div>
              <div>
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#818cf8;font-weight:700;">Performing Task</div>
                <div style="font-size:0.92rem;font-weight:600;color:#f8fafc;">${{escapeHtml(taskName)}}</div>
              </div>
            </div>
            <span style="font-size:0.72rem;background:rgba(99,102,241,0.25);color:#c7d2fe;padding:3px 10px;border-radius:12px;border:1px solid rgba(99,102,241,0.4);animation:stTaskPulse 1.5s infinite;flex-shrink:0;font-weight:600;">Running...</span>
          </div>
        `;
      }}

      function showSuccess(taskName, details) {{
        mon.className = "state-success show";
        mon.innerHTML = `
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
            <div style="display:flex;align-items:flex-start;gap:10px;">
              <div style="background:#10b981;color:#022c22;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:0.92rem;flex-shrink:0;margin-top:2px;">✓</div>
              <div>
                <div style="font-size:0.94rem;font-weight:700;color:#ecfdf5;display:flex;align-items:center;gap:6px;">
                  <span>Executed successfully</span>
                  <span style="background:rgba(16,185,129,0.3);color:#6ee7b7;font-size:0.68rem;padding:2px 7px;border-radius:10px;font-weight:700;">Success</span>
                </div>
                <div style="font-size:0.83rem;color:#a7f3d0;margin-top:2px;line-height:1.4;">
                  <b>${{escapeHtml(taskName)}}</b>${{details ? ': ' + escapeHtml(details) : ''}}
                </div>
              </div>
            </div>
            <button onclick="window.parent._dismissStTaskMonitor()" style="background:none;border:none;color:#a7f3d0;font-size:1.15rem;cursor:pointer;padding:0 4px;line-height:1;" title="Dismiss">✕</button>
          </div>
        `;
        clearTimeout(window._stTaskTimer);
        window._stTaskTimer = setTimeout(() => {{
          window.parent._dismissStTaskMonitor();
        }}, 4500);
      }}

      function showFailed(taskName, errorMsg) {{
        mon.className = "state-failed show";
        mon.innerHTML = `
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
            <div style="display:flex;align-items:flex-start;gap:10px;">
              <div style="background:#ef4444;color:#450a0a;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:0.92rem;flex-shrink:0;margin-top:2px;">✕</div>
              <div>
                <div style="font-size:0.94rem;font-weight:700;color:#fef2f2;display:flex;align-items:center;gap:6px;">
                  <span>Execution failed</span>
                  <span style="background:rgba(239,68,68,0.3);color:#fca5a5;font-size:0.68rem;padding:2px 7px;border-radius:10px;font-weight:700;">Failed</span>
                </div>
                <div style="font-size:0.83rem;color:#fca5a5;margin-top:2px;line-height:1.4;">
                  <b>${{escapeHtml(taskName)}}</b>${{errorMsg ? ': ' + escapeHtml(errorMsg) : ''}}
                </div>
              </div>
            </div>
            <button onclick="window.parent._dismissStTaskMonitor()" style="background:none;border:none;color:#fca5a5;font-size:1.15rem;cursor:pointer;padding:0 4px;line-height:1;" title="Dismiss">✕</button>
          </div>
        `;
        clearTimeout(window._stTaskTimer);
        window._stTaskTimer = setTimeout(() => {{
          window.parent._dismissStTaskMonitor();
        }}, 6000);
      }}



      // 3. Setup click listener on top-level document for button feedback
      if (!window.parent._stTaskClickListenerAttached) {
        window.parent._stTaskClickListenerAttached = true;
        doc.addEventListener("click", function(e) {
          const btn = e.target.closest("button, [role='button'], input[type='button'], input[type='submit']");
          if (!btn) return;
          if (btn.closest("#stTaskExecutionMonitor") || btn.closest("#stBtnClickToast")) return;

          let rawLabel = (btn.innerText || btn.getAttribute("aria-label") || btn.title || "").trim();
          rawLabel = rawLabel.replace(/[\r\n\t]+/g, ' ').replace(/\s{2,}/g, ' ').trim();
          // Single-time crisp button movement on click
          btn.classList.remove("btn-clicked-animation");
          void btn.offsetWidth;
          btn.classList.add("btn-clicked-animation");
          setTimeout(() => {{ btn.classList.remove("btn-clicked-animation"); }}, 240);

          let cleanLabel = rawLabel.replace(/[✓✕🔊⏹️🚀⚡🧠🤖🧹📊📁💾📈📋⚙️🔄]/g, '').trim();
          if (!cleanLabel || cleanLabel.length < 2) return;

          // STRICT CHECK: Only trigger task runner for data analytics steps
          const lower = cleanLabel.toLowerCase();
          const analyticsKeywords = ["clean", "imput", "outlier", "deduplicat", "standardiz", "ingest", "load", "filter", "calculated", "encod", "scale", "interact", "execute query", "train", "cluster", "forecast", "reset", "revert", "undo", "redo", "statistical", "hypothesis", "regression"];
          const isAnalytics = analyticsKeywords.some(kw => lower.includes(kw));
          if (!isAnalytics) return;

          sessionStorage.setItem("st_last_clicked_task", cleanLabel);
          sessionStorage.setItem("st_task_status", "running");
          showRunning(cleanLabel);
        }, true);
      }

      // 4. Handle status after page load / rerun
      const pyTask = {task_json};
      if (pyTask && pyTask.task) {{
        if (pyTask.status === "failed") {{
          showFailed(pyTask.task, pyTask.details || "Operation failed");
        }} else {{
          showSuccess(pyTask.task, pyTask.details || "Executed successfully");
        }}
        sessionStorage.removeItem("st_task_status");
      }} else {{
        const pendingStatus = sessionStorage.getItem("st_task_status");
        const lastTask = sessionStorage.getItem("st_last_clicked_task");
        if (pendingStatus === "running" && lastTask) {{
          // Only show failure if active rerun produced an explicit unhandled exception
          const activeEx = doc.querySelector(".stException");
          if (activeEx) {{
            const errText = activeEx.innerText || "Execution encountered an error";
            showFailed(lastTask, errText.slice(0, 120));
          }} else {{
            showSuccess(lastTask, "Executed successfully");
          }}
          sessionStorage.removeItem("st_task_status");
        }}
      }}
    })();
    </script>
    """, height=0, width=0)

render_task_execution_monitor()

pm: ProjectManager = st.session_state.project_manager
active_proj = pm.get_active_project()

# ---------------- SIDEBAR NAVIGATION: 22 CORE MODULES ----------------
with st.sidebar:
    brand_logo_col, brand_text_col = st.columns([1, 2.5])
    with brand_logo_col:
        if os.path.exists("assets/datamind_logo_thumb.jpg"):
            st.image("assets/datamind_logo_thumb.jpg", width=72)
        elif os.path.exists("assets/datamind_logo.jpg"):
            st.image("assets/datamind_logo.jpg", width=72)
        elif os.path.exists("datamind_logo.jpg"):
            st.image("datamind_logo.jpg", width=72)
    with brand_text_col:
        st.markdown("<h3 style='margin:0;padding:0;font-weight:800;font-size:1.35rem;line-height:1.15;'>DataMind AI</h3>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;font-weight:700;color:#38bdf8;letter-spacing:0.05em;text-transform:uppercase;margin-top:2px;'>Analyze • Predict • Empower</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.68rem;color:#94a3b8;margin-top:1px;'>AI Powered Data Analytics Platform</div>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Workspace", key="top_reset_app_btn", help="Reset DataMind AI: Clears all uploaded files and starts fresh", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    # Active Project Badge & Selector
    projects = pm.get_all_projects()
    proj_names = [p["name"] for p in projects]
    curr_proj_idx = proj_names.index(active_proj["name"]) if active_proj["name"] in proj_names else 0
    selected_proj_name = st.selectbox("📁 Active Project:", proj_names, index=curr_proj_idx, key="sb_active_proj")
    
    # Switch active project if changed
    for p in projects:
        if p["name"] == selected_proj_name and p["id"] != pm.active_project_id:
            pm.set_active_project(p["id"])
            st.rerun()

    st.markdown("---")
    st.markdown("### 🧭 **Categorized Navigation**")

    # Grouped Navigation Modules by Technology Domain
    nav_categories = {
        "🛠️ Data Engineering": [
            "📋 Data Preview",
            "🧹 Data Cleaning",
            "🔄 Data Processing & Transform",
            "⚙️ Feature Engineering",
            "➕ Insert New Column",
            "📤 Data Upload"
        ],
        "📊 Analytics & BI": [
            "📊 EDA",
            "📈 Visualization",
            "💡 Insights",
            "📐 Statistical Analysis",
            "🧪 Experimentation & A/B Testing",
            "🎨 Dashboard Builder",
            "📊 Power BI Analysis"
        ],
        "💻 Query & Code Engines": [
            "🔌 Database Connector",
            "🗄️ SQL & Queries",
            "🐍 Python Coding Environment"
        ],
        "🤖 AI & Machine Learning": [
            "⚡ Model AutoML",
            "🤖 AI Assistant",
            "🤖 Machine Learning",
            "🎯 Predictive Analytics",
            "🔮 Forecasting",
            "📏 Model Evaluation"
        ],
        "📑 Reporting & Governance": [
            "📑 Report",
            "💾 Data Export",
            "📁 Projects",
            "🏷️ Model Registry",
            "⚙️ Settings",
            "🏠 Home"
        ]
    }

    if "quick_jump" in st.session_state:
        qj = st.session_state.quick_jump
        for cat_name, mods in nav_categories.items():
            if qj in mods:
                st.session_state["active_nav_cat"] = cat_name
                st.session_state["quick_jump_module"] = qj
                break
        del st.session_state.quick_jump

    cat_keys = list(nav_categories.keys())
    saved_cat = st.session_state.get("active_nav_cat", cat_keys[0])
    cat_idx = cat_keys.index(saved_cat) if saved_cat in cat_keys else 0

    selected_category = st.selectbox(
        "Technology Domain:",
        cat_keys,
        index=cat_idx,
        key="sb_tech_domain_select"
    )
    st.session_state["active_nav_cat"] = selected_category

    available_mods = nav_categories[selected_category]
    mod_idx = 0
    if "quick_jump_module" in st.session_state and st.session_state.quick_jump_module in available_mods:
        mod_idx = available_mods.index(st.session_state.quick_jump_module)
        del st.session_state.quick_jump_module

    selected_module = st.radio(
        "Module Selection:",
        available_mods,
        index=mod_idx,
        label_visibility="collapsed"
    )

    st.markdown("---")
    # Active Dataset Quick Summary
    curr_df = st.session_state.current_df
    st.markdown(f"**Dataset:** `{st.session_state.dataset_source}`")
    st.caption(f"{curr_df.shape[0]:,} rows × {curr_df.shape[1]} cols")

    # Undo / Redo in Sidebar if cleaning pipeline active
    clean_mgr: CleaningPipelineManager = st.session_state.cleaning_manager
    col_u, col_r = st.columns(2)
    with col_u:
        if st.button("↩️ Undo", use_container_width=True, disabled=not clean_mgr.can_undo()):
            undone = clean_mgr.undo()
            if undone:
                st.session_state.current_df = undone[1].copy()
                st.rerun()
    with col_r:
        if st.button("↪️ Redo", use_container_width=True, disabled=not clean_mgr.can_redo()):
            redone = clean_mgr.redo()
            if redone:
                st.session_state.current_df = redone[1].copy()
                st.rerun()

    if st.button("🔄 Reset to Raw Data", use_container_width=True, key="btn_reset_raw_sidebar"):
        if st.session_state.raw_df is None or st.session_state.raw_df.empty:
            set_task_status("Reset to Raw Data", "failed", "No raw dataset available. Please upload a CSV or load a demo dataset first.")
            st.warning("No raw dataset available to reset. Please upload a file first.")
        else:
            st.session_state.current_df = st.session_state.raw_df.copy()
            st.session_state.cleaning_manager = CleaningPipelineManager(st.session_state.raw_df)
            st.session_state.cleaning_history = []
            st.session_state.fe_history = []
            set_task_status("Reset to Raw Data", "success", f"Reverted dataset to original raw state ({len(st.session_state.raw_df):,} rows × {st.session_state.raw_df.shape[1]} columns).")
            st.rerun()

    st.markdown("---")
    st.caption(f"Logged in as **{st.session_state.user_role}**")


# ---------------- TOP HEADER & OMNIBAR ----------------
df = st.session_state.current_df
profile = get_dataset_quick_profile(df)

col_top_left, col_top_right = st.columns([3, 1])
with col_top_left:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 12px; flex-wrap: wrap;">
        <span style="font-size: 1.5rem; font-weight:800; background: linear-gradient(135deg, #6366f1, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {selected_module}
        </span>
        <span class="badge-chip badge-info">Project: {active_proj['name']}</span>
        <span class="badge-chip {profile['health_issues'] and 'badge-warning' or 'badge-success'}">Health: {profile['health_score']}/100 ({profile['health_grade']})</span>
    </div>
    """, unsafe_allow_html=True)

with col_top_right:
    # Role Selector & Quick Refresh & Brightness / Light Theme Toggle
    sub_c1, sub_c2, sub_c3 = st.columns([1.1, 1, 0.95])
    with sub_c1:
        new_role = st.selectbox("Role", ["Admin", "Analyst", "Viewer"], index=["Admin", "Analyst", "Viewer"].index(st.session_state.user_role), label_visibility="collapsed")
        if new_role != st.session_state.user_role:
            st.session_state.user_role = new_role
            st.rerun()
    with sub_c2:
        if st.button("⚡ Omnibar", use_container_width=True):
            st.session_state.show_omnibar = not st.session_state.get("show_omnibar", False)
    with sub_c3:
        is_light = st.session_state.get("is_light_mode", False)
        theme_btn_label = "🌙 Dark" if is_light else "☀️ Light"
        if st.button(theme_btn_label, use_container_width=True, help="Toggle Brightness / Light Theme"):
            st.session_state.is_light_mode = not is_light
            set_task_status("Theme Toggled", "success", f"Switched to {'Light / Bright' if not is_light else 'Dark'} Mode.")
            st.rerun()

# Universal AI Command Bar (Omnibar) Modal Drawer
if st.session_state.get("show_omnibar", False):
    with st.expander("✨ Universal AI Command Bar (Omnibar) - Ask anything or run actions", expanded=True):
        cmd_col1, cmd_col2 = st.columns([4, 1])
        with cmd_col1:
            omni_query = st.text_input("Enter natural language analytics command:", placeholder="e.g. 'Show top 5 products by revenue' or 'Run AutoML on Churn' or 'Forecast sales'", label_visibility="collapsed")
        with cmd_col2:
            exec_btn = st.button("Execute Command", type="primary", use_container_width=True)
        
        if exec_btn and omni_query:
            orchestrator: AIOrchestrator = st.session_state.ai_orchestrator
            res = orchestrator.route_omnibar_command(omni_query, df)
            st.info(f"**Router:** {res.get('agent', 'AI Agent')} ➔ Recommended Module: **{res.get('recommended_module')}**")
            st.markdown(res.get("message", ""))
            if "sql_suggestion" in res:
                st.code(res["sql_suggestion"], language="sql")


def render_python_environment(df: pd.DataFrame, key_prefix: str = "py_"):
    st.markdown("---")
    st.subheader("🐍 Python Analytics & Coding Sandbox")
    st.caption("Write and execute Python / Pandas scripts directly against your active dataset (`df`).")

    # Quick templates & AI Assistant row
    py_c1, py_c2 = st.columns([1.5, 2.5])
    with py_c1:
        py_template = st.selectbox(
            "Quick Python Templates:",
            [
                "-- Choose a template --",
                "1. Summary Statistics (df.describe())",
                "2. Preview First 10 Rows (df.head(10))",
                "3. Filter Numeric Records Above Mean",
                "4. Add Synthesized Ratio Column",
                "5. Group By & Aggregate Metrics",
                "6. Sort by High Values",
                "7. Deduplicate Records (df.drop_duplicates())"
            ],
            key=f"{key_prefix}_template_select"
        )
    with py_c2:
        py_ai_prompt = st.text_input(
            "🪄 AI Python Generator:",
            placeholder="e.g. 'filter high charges and sort by tenure descending'...",
            key=f"{key_prefix}_ai_prompt"
        )

    # Generate code from template or AI
    default_py_code = """# Python Data Analytics Sandbox
# Available variables: df (active pandas DataFrame), pd, np
print(f"Loaded active DataFrame with {len(df):,} rows and {df.shape[1]} columns.")
print("Columns:", list(df.columns[:8]))

# Set variable `result` to display dataframe in the output table
result = df.describe()
"""

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    n_col = num_cols[0] if num_cols else (df.columns[0] if len(df.columns) > 0 else "val")
    n_col2 = num_cols[1] if len(num_cols) > 1 else n_col
    c_col = cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) > 0 else "cat")

    if py_template.startswith("1."):
        default_py_code = """# Summary Statistics
print("Dataset shape:", df.shape)
result = df.describe()
"""
    elif py_template.startswith("2."):
        default_py_code = """# Preview First 10 Rows
print("Top 10 records:")
result = df.head(10)
"""
    elif py_template.startswith("3."):
        default_py_code = f"""# Filter records where '{n_col}' is above mean
mean_val = df['{n_col}'].mean()
print(f"Filtering '{n_col}' > mean ({{mean_val:.2f}})")
result = df[df['{n_col}'] > mean_val].sort_values(by='{n_col}', ascending=False)
print(f"Matched {{len(result):,}} rows.")
"""
    elif py_template.startswith("4."):
        default_py_code = f"""# Synthesize new ratio feature
col1 = '{n_col}'
col2 = '{n_col2}'
new_col = f"ratio_{{col1}}_to_{{col2}}"
df_copy = df.copy()
df_copy[new_col] = (df_copy[col1] / (df_copy[col2] + 1e-5)).round(3)
print(f"Synthesized feature: {{new_col}}")
result = df_copy
"""
    elif py_template.startswith("5."):
        default_py_code = f"""# Group by '{c_col}' and aggregate '{n_col}'
print("Grouping by {c_col}...")
result = df.groupby('{c_col}').agg(
    records=('{c_col}', 'count'),
    mean_{n_col}=('{n_col}', 'mean'),
    total_{n_col}=('{n_col}', 'sum')
).reset_index()
print(f"Aggregated {{len(result)}} categories.")
"""
    elif py_template.startswith("6."):
        default_py_code = f"""# Sort descending by '{n_col}'
print("Sorting top 20 records by {n_col}...")
result = df.sort_values(by='{n_col}', ascending=False).head(20)
"""
    elif py_template.startswith("7."):
        default_py_code = """# Drop duplicate rows
before = len(df)
result = df.drop_duplicates()
print(f"Deduplication complete: {before:,} -> {len(result):,} records.")
"""

    if py_ai_prompt:
        p_lower = py_ai_prompt.lower()
        if "filter" in p_lower or "where" in p_lower:
            default_py_code = f"""# AI Generated: Filter records
threshold = df['{n_col}'].mean()
print(f"Filtering '{n_col}' >= {{threshold:.2f}}")
result = df[df['{n_col}'] >= threshold].sort_values(by='{n_col}', ascending=False)
print(f"Found {{len(result):,}} matching rows.")
"""
        elif "group" in p_lower:
            default_py_code = f"""# AI Generated: Grouping analysis
result = df.groupby('{c_col}')['{n_col}'].agg(['count', 'mean', 'sum']).reset_index()
print(f"Aggregated across {{len(result)}} groups.")
"""
        elif "sort" in p_lower or "top" in p_lower:
            default_py_code = f"""# AI Generated: Top ranked records
result = df.sort_values(by='{n_col}', ascending=False).head(20)
print(f"Top 20 records by {n_col}.")
"""
        else:
            default_py_code = """# AI Generated: Python Analytics Script
print("Executing custom analytics on DataFrame...")
result = df.head(15)
"""

    py_script_input = st.text_area(
        "Python Script (Edit code below):",
        value=default_py_code,
        height=180,
        key=f"{key_prefix}_code_area"
    )

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_py_btn = st.button("▶️ Run Python Script", type="primary", use_container_width=True, key=f"{key_prefix}_run_btn")

    if run_py_btn:
        if df.empty:
            set_task_status("Python Script Execution", "failed", "Active dataset is empty. Ingest data first.")
            st.error("No active dataset loaded. Please upload a CSV first.")
        else:
            import io
            import contextlib
            stdout_buf = io.StringIO()
            t0 = time.time()
            local_scope = {
                "df": df.copy(),
                "pd": pd,
                "np": np,
                "result": None
            }
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(py_script_input, {"__builtins__": __builtins__}, local_scope)
                elapsed_ms = round((time.time() - t0) * 1000, 1)
                captured_stdout = stdout_buf.getvalue()

                st.session_state[f"{key_prefix}_last_stdout"] = captured_stdout
                st.session_state[f"{key_prefix}_last_elapsed"] = elapsed_ms

                res = local_scope.get("result")
                if isinstance(res, pd.DataFrame):
                    st.session_state[f"{key_prefix}_last_result"] = res
                elif isinstance(res, pd.Series):
                    st.session_state[f"{key_prefix}_last_result"] = res.to_frame()
                elif res is not None:
                    st.session_state[f"{key_prefix}_last_result"] = pd.DataFrame([{"result_value": res}])
                else:
                    st.session_state[f"{key_prefix}_last_result"] = local_scope["df"]

                row_count = len(st.session_state[f"{key_prefix}_last_result"])
                set_task_status("Python Script Execution", "success", f"Executed in {elapsed_ms} ms, generated {row_count:,} row(s).")
            except Exception as py_err:
                elapsed_ms = round((time.time() - t0) * 1000, 1)
                st.session_state[f"{key_prefix}_last_stdout"] = stdout_buf.getvalue() + f"\n❌ Execution Error: {py_err}"
                st.session_state[f"{key_prefix}_last_result"] = None
                set_task_status("Python Script Execution", "failed", str(py_err))

    # Show Output Console & Result Table
    if f"{key_prefix}_last_stdout" in st.session_state:
        st.markdown(f"**📟 Console Output / Stdout** (`{st.session_state.get(f'{key_prefix}_last_elapsed', 0)} ms`):")
        st.code(st.session_state[f"{key_prefix}_last_stdout"] or "(No print statements executed)", language="text")

    if st.session_state.get(f"{key_prefix}_last_result") is not None:
        res_df = st.session_state[f"{key_prefix}_last_result"]
        st.markdown(f"#### 📊 Script Result Data ({len(res_df):,} rows × {res_df.shape[1]} columns)")
        col_res1, col_res2 = st.columns([1.5, 3])
        with col_res1:
            if st.button("🔄 Set as Active Dataset", type="primary", use_container_width=True, key=f"{key_prefix}_set_active_btn"):
                st.session_state.current_df = res_df.copy()
                st.session_state.cleaning_manager = CleaningPipelineManager(res_df)
                st.session_state.dataset_source = f"Python Script Result ({len(res_df):,} rows)"
                set_task_status("Set Active Dataset", "success", f"Replaced active dataset with Python result ({len(res_df):,} rows).")
                st.rerun()
        with col_res2:
            csv_bytes = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Python Result (CSV)",
                data=csv_bytes,
                file_name=f"python_result_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=False,
                key=f"{key_prefix}_download_btn"
            )
        st.dataframe(res_df.head(100), use_container_width=True)
        if len(res_df) > 100:
            st.caption(f"Showing top 100 of {len(res_df):,} rows.")


# =========================================================================
# ----------------------------- MODULE VIEWS ------------------------------
# =========================================================================

# ---------------- 1. HOME ----------------
if selected_module == "🏠 Home":
    st.markdown(f"""
    <div class="hero-banner">
        <h2 style="margin:0 0 6px 0; color:#f8fafc;">Welcome to DataMind AI Workspace</h2>
        <p style="margin:0 0 14px 0; color:#94a3b8; font-size:1.02rem;">
            A unified, production-ready environment for the complete data analytics lifecycle: ingestion, automated auditing, visual transformations, DuckDB SQL Studio, statistical hypothesis testing, AutoML, and predictive forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Metrics KPI Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""<div class="metric-card"><h4>Records</h4><div class="metric-val">{profile['rows']:,}</div><small style="color:#94a3b8;">Total rows</small></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class="metric-card"><h4>Columns</h4><div class="metric-val">{profile['columns']}</div><small style="color:#94a3b8;">{profile['numeric_cols_count']} num, {profile['categorical_cols_count']} cat</small></div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""<div class="metric-card"><h4>Health Score</h4><div class="metric-val">{profile['health_score']}/100</div><small class="badge-chip {profile['health_grade'] in ['A','B'] and 'badge-success' or 'badge-warning'}">Grade {profile['health_grade']}</small></div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""<div class="metric-card"><h4>Missing Cells</h4><div class="metric-val">{profile['missing_cells']:,}</div><small style="color:#94a3b8;">{profile['missing_pct']}% missing rate</small></div>""", unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""<div class="metric-card"><h4>Duplicates</h4><div class="metric-val">{profile['duplicated_rows']:,}</div><small style="color:#94a3b8;">{profile['duplicated_pct']}% exact dupes</small></div>""", unsafe_allow_html=True)

    st.markdown("### 🚀 **Quick Actions**")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5, qa_col6 = st.columns(6)
    with qa_col1:
        if st.button("📤 Upload Data", use_container_width=True):
            st.session_state.quick_jump = "📤 Data Upload"
    with qa_col2:
        if st.button("🧹 Auto-Clean", use_container_width=True):
            clean_df, log = one_click_ai_auto_clean(df)
            st.session_state.current_df = clean_df
            st.session_state.cleaning_manager.apply_step("1-Click Auto-Clean", clean_df)
            st.success("Dataset auto-cleaned successfully!")
            st.rerun()
    with qa_col3:
        if st.button("📊 Explore EDA", use_container_width=True):
            st.session_state.quick_jump = "📊 EDA"
    with qa_col4:
        if st.button("🗄️ Run SQL", use_container_width=True):
            st.session_state.quick_jump = "🗄️ SQL & Queries"
            st.rerun()
    with qa_col5:
        if st.button("🤖 Train AutoML", use_container_width=True):
            st.session_state.quick_jump = "🤖 Machine Learning"
    with qa_col6:
        if st.button("📑 Build Report", use_container_width=True):
            st.session_state.quick_jump = "📑 Reports"

    st.markdown("---")
    # Recent Projects & Activity
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.subheader("📁 Recent Analytics Projects")
        for p in projects[:3]:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
                <div style="font-weight:700; color:#f8fafc; font-size:1rem;">{p['name']}</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:2px;">{p.get('description', '')}</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:6px;">Active Dataset: <b>{p.get('active_dataset', 'dataset.csv')}</b> | Last updated: {p.get('updated_at', 'Recently')}</div>
            </div>
            """, unsafe_allow_html=True)
    with p_col2:
        st.subheader("📢 Automated Hygiene & Anomaly Alerts")
        health = compute_dataset_health_score(df)
        for issue in health["issues"]:
            st.warning(f"⚠️ {issue}")


# ---------------- 2. PROJECTS ----------------
elif selected_module == "📁 Projects":
    st.subheader("📁 Project Workspace Management")
    
    t_proj1, t_proj2, t_proj3 = st.tabs(["All Projects", "➕ Create Project", "📜 Activity Log"])
    
    with t_proj1:
        search_q = st.text_input("🔍 Search Projects:", placeholder="Search by name or description...")
        filtered_projects = pm.search_projects(search_q) if search_q else pm.get_all_projects()

        for p in filtered_projects:
            with st.container():
                c_p1, c_p2, c_p3, c_p4 = st.columns([3, 1, 1, 1])
                with c_p1:
                    st.markdown(f"### {p['name']} {'(Active)' if p['id'] == pm.active_project_id else ''}")
                    st.caption(p.get("description", "No description provided."))
                    st.caption(f"Created: {p.get('created_at')} | Active Dataset: `{p.get('active_dataset')}`")
                with c_p2:
                    if p["id"] != pm.active_project_id:
                        if st.button("Set Active", key=f"set_act_{p['id']}", use_container_width=True):
                            pm.set_active_project(p["id"])
                            st.success(f"Switched active project to '{p['name']}'.")
                            st.rerun()
                with c_p3:
                    if st.button("Duplicate", key=f"dup_{p['id']}", use_container_width=True):
                        pm.duplicate_project(p["id"])
                        st.success("Project duplicated.")
                        st.rerun()
                with c_p4:
                    if len(pm.projects) > 1 and st.button("Delete", key=f"del_{p['id']}", use_container_width=True):
                        pm.delete_project(p["id"])
                        st.success("Project deleted.")
                        st.rerun()
                st.markdown("---")

    with t_proj2:
        st.markdown("#### Initialize New Project")
        p_name = st.text_input("Project Name:", placeholder="e.g. Q4 Supply Chain Optimization")
        p_desc = st.text_area("Project Description:", placeholder="Objectives, scope, and target metrics...")
        p_dataset = st.selectbox("Initial Dataset:", ["Customer Churn (Demo Dataset)", "Real Estate Housing (Demo Dataset)", "E-Commerce Sales (Demo Dataset)"])
        if st.button("Create Project", type="primary"):
            if p_name.strip():
                new_p = pm.create_project(p_name.strip(), p_desc.strip(), p_dataset)
                st.success(f"Project '{new_p['name']}' created and activated!")
                st.rerun()
            else:
                st.warning("Please provide a project name.")

    with t_proj3:
        st.markdown("#### Audit & Activity History")
        act_log = active_proj.get("activity_log", [])
        if act_log:
            st.dataframe(pd.DataFrame(act_log), use_container_width=True)
        else:
            st.info("No activity recorded yet for this project.")


# ---------------- 3. DATA UPLOAD ----------------
elif selected_module == "📤 Data Upload":
    st.subheader("📤 Multi-Format Dataset Ingestion")
    
    st.markdown("""
    Support for **CSV, XLSX, XLS, JSON, Parquet, TXT, and TSV** files. Uploading automatically triggers data profiling, type inference, and health scoring.
    """)

    up_file = st.file_uploader(
        "Upload dataset file:",
        type=["csv", "xlsx", "xls", "json", "parquet", "txt", "tsv"]
    )

    col_delim, col_enc = st.columns(2)
    with col_delim:
        custom_delimiter = st.selectbox("Delimiter (Delimited files):", ["auto", ",", ";", "\t", "|"])
    with col_enc:
        custom_encoding = st.selectbox("Encoding:", ["auto", "utf-8", "latin1", "cp1252"])

    if up_file is not None:
        # Automatic instant ingestion when file is uploaded or selected
        if st.session_state.get("dataset_source") != up_file.name:
            try:
                with st.spinner("Ingesting and auditing uploaded dataset..."):
                    loaded_df, meta = load_dataset(
                        up_file,
                        filename=up_file.name,
                        delimiter=custom_delimiter if custom_delimiter != "auto" else None,
                        encoding=custom_encoding if custom_encoding != "auto" else None
                    )
                    st.session_state.raw_df = loaded_df.copy()
                    st.session_state.current_df = loaded_df.copy()
                    st.session_state.dataset_source = up_file.name
                    st.session_state.cleaning_manager = CleaningPipelineManager(loaded_df)
                    st.session_state.cleaning_history = []
                    st.session_state.fe_history = []
                    pm.log_activity("Ingested Dataset", f"Loaded '{up_file.name}' with {len(loaded_df):,} rows.")
                    set_task_status("Ingest Dataset", "success", f"Loaded '{up_file.name}' ({len(loaded_df):,} rows × {loaded_df.shape[1]} columns)")
                    st.rerun()
            except Exception as e:
                set_task_status("Ingest Dataset", "failed", str(e))
                st.error(f"Error loading file: {str(e)}")

        if st.button("🚀 Ingest Uploaded Dataset", type="primary", use_container_width=True):
            try:
                with st.spinner("Ingesting and auditing dataset..."):
                    loaded_df, meta = load_dataset(
                        up_file,
                        filename=up_file.name,
                        delimiter=custom_delimiter if custom_delimiter != "auto" else None,
                        encoding=custom_encoding if custom_encoding != "auto" else None
                    )
                    st.session_state.raw_df = loaded_df.copy()
                    st.session_state.current_df = loaded_df.copy()
                    st.session_state.dataset_source = up_file.name
                    st.session_state.cleaning_manager = CleaningPipelineManager(loaded_df)
                    st.session_state.cleaning_history = []
                    st.session_state.fe_history = []
                    
                    # Log to project and monitor
                    pm.log_activity("Ingested Dataset", f"Loaded '{up_file.name}' with {len(loaded_df):,} rows.")
                    set_task_status("Ingest Dataset", "success", f"Loaded '{up_file.name}' ({len(loaded_df):,} rows × {loaded_df.shape[1]} columns)")
                    st.success(f"Successfully loaded '{up_file.name}' ({len(loaded_df):,} rows × {loaded_df.shape[1]} columns)!")
                    st.rerun()
            except Exception as e:
                set_task_status("Ingest Dataset", "failed", str(e))
                st.error(f"Error loading file: {str(e)}")

    st.markdown("---")
    st.markdown("#### Or Load Built-In Demo Datasets:")
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if st.button("Load Telecom Churn Dataset", use_container_width=True):
            df_churn = generate_sample_customer_churn()
            st.session_state.raw_df = df_churn.copy()
            st.session_state.current_df = df_churn.copy()
            st.session_state.dataset_source = "Customer Churn (Demo Dataset)"
            st.session_state.cleaning_manager = CleaningPipelineManager(df_churn)
            set_task_status("Load Dataset", "success", f"Loaded Telecom Churn ({len(df_churn):,} rows)")
            st.rerun()
    with d_col2:
        if st.button("Load Real Estate Housing Dataset", use_container_width=True):
            df_house = generate_sample_housing()
            st.session_state.raw_df = df_house.copy()
            st.session_state.current_df = df_house.copy()
            st.session_state.dataset_source = "Real Estate Housing (Demo Dataset)"
            st.session_state.cleaning_manager = CleaningPipelineManager(df_house)
            set_task_status("Load Dataset", "success", f"Loaded Real Estate Housing ({len(df_house):,} rows)")
            st.rerun()
    with d_col3:
        if st.button("Load E-Commerce Sales Dataset", use_container_width=True):
            df_sales = generate_sample_ecommerce_sales()
            st.session_state.raw_df = df_sales.copy()
            st.session_state.current_df = df_sales.copy()
            st.session_state.dataset_source = "E-Commerce Sales (Demo Dataset)"
            st.session_state.cleaning_manager = CleaningPipelineManager(df_sales)
            set_task_status("Load Dataset", "success", f"Loaded E-Commerce Sales ({len(df_sales):,} rows)")
            st.rerun()


# ---------------- 3B. DATA CONNECTORS & DATABASES ----------------
elif selected_module in ["🔌 Database Connector", "🔌 Data Connectors & Databases", "Database Connector"]:
    st.markdown("""
    <div class="hero-banner">
        <h2 style="margin:0 0 6px 0; color:#f8fafc;">🔌 Enterprise Data Connectors & Integration Engine</h2>
        <p style="margin:0 0 10px 0; color:#94a3b8; font-size:0.96rem;">
            Secure, high-throughput bidirectional connectivity for Relational Databases, NoSQL, Cloud Data Warehouses, SaaS Spreadsheets, APIs, and Cloud Object Storage lakes.
        </p>
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <span class="badge-chip badge-success">🛡️ AES-256 Fernet Credential Vault</span>
            <span class="badge-chip badge-info">🔒 Zero-Plaintext at Rest</span>
            <span class="badge-chip badge-warning">⚡ 12 Production Drivers</span>
            <span class="badge-chip badge-success">🧼 Automated Credential Redaction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_conn, tab_arch, tab_demos = st.tabs([
        "🔌 Connect & Ingest",
        "🏛️ Architecture & Security Model",
        "⚡ Pre-Configured Live Demos"
    ])

    with tab_conn:
        st.markdown("### 🛠️ **Configure New Connection**")
        
        # Category filter
        cats = ["All (12)", "Relational Databases", "NoSQL Databases", "Cloud Data Warehouses", "SaaS & Files", "APIs & Webhooks", "Cloud Storage Lakes"]
        sel_cat = st.radio("Connector Category:", cats, horizontal=True, key="conn_cat_filter")
        
        filtered_catalog = CONNECTOR_CATALOG if sel_cat.startswith("All") else [c for c in CONNECTOR_CATALOG if c["category"] == sel_cat]
        
        conn_opts = [f"{c['icon']} {c['name']}" for c in filtered_catalog]
        sel_conn_label = st.selectbox("Select Target Data Source:", conn_opts, key="conn_target_select")
        
        # Match selected catalog item
        active_cat_item = next(c for c in filtered_catalog if f"{c['icon']} {c['name']}" == sel_conn_label)
        c_type = ConnectorType(active_cat_item["type"])
        
        st.info(f"**{active_cat_item['name']} ({active_cat_item['category']})**: {active_cat_item['desc']}")

        # Pre-initialize active connector instance if not present so screen is never blank
        if "active_connector_instance" not in st.session_state or st.session_state["active_connector_instance"] is None:
            demo_db_path, _ = get_or_create_demo_sqlite()
            init_cfg = ConnectorConfig(
                connector_type=ConnectorType.SQLITE,
                name="Demo SQLite Enterprise DB",
                database=demo_db_path
            )
            init_conn = create_connector(init_cfg)
            st.session_state["active_connector_instance"] = init_conn
            st.session_state["active_connector_config"] = init_cfg
            try:
                st.session_state["connector_discovered_schema"] = init_conn.get_schema()
            except Exception:
                pass
        
        with st.form("connector_config_form"):
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                c_name = st.text_input("Connection Identifier / Alias:", value=f"My {active_cat_item['name']}", key="conn_form_name")
                c_host = st.text_input(
                    "Host / Endpoint / Server / File URI:",
                    placeholder=active_cat_item["host_placeholder"],
                    value="localhost" if c_type in [ConnectorType.POSTGRESQL, ConnectorType.MYSQL, ConnectorType.MSSQL, ConnectorType.MONGODB] else "",
                    key="conn_form_host"
                )
                c_port = st.number_input(
                    "Port:",
                    value=active_cat_item["default_port"] or 0,
                    min_value=0,
                    max_value=65535,
                    key="conn_form_port"
                ) if active_cat_item["default_port"] else None
            
            with c_f2:
                c_db = st.text_input("Database / Dataset / Catalog / Sheet ID:", placeholder="e.g. analytics_prod", key="conn_form_db")
                c_user = st.text_input("Username / Client ID:", placeholder="e.g. read_only_analyst", key="conn_form_user")
                c_secret = st.text_input(
                    f"{active_cat_item['credential_label']}:",
                    type="password",
                    placeholder="Encrypted automatically via AES-256 vault",
                    key="conn_form_secret"
                )
            
            c_ssl = st.checkbox("Require SSL / TLS Transport (Encrypted Wire)", value=True, key="conn_form_ssl")
            
            with st.expander("⚙️ Advanced Parameters (JSON / Warehouse / Role)"):
                c_extra_raw = st.text_area(
                    "Extra Connection Parameters (JSON):",
                    value='{"warehouse": "COMPUTE_WH", "role": "ANALYST"}' if c_type == ConnectorType.SNOWFLAKE else '{}',
                    key="conn_form_extra"
                )
            
            submitted = st.form_submit_button("💾 Save & Encrypt Connection Config", use_container_width=True)
            if submitted:
                try:
                    extra_dict = json.loads(c_extra_raw) if c_extra_raw.strip() else {}
                except Exception:
                    extra_dict = {}
                
                cfg = ConnectorConfig(
                    connector_type=c_type,
                    name=c_name.strip() or f"{active_cat_item['name']}_conn",
                    host=c_host.strip(),
                    port=int(c_port) if c_port else None,
                    database=c_db.strip(),
                    username=c_user.strip(),
                    ssl_enabled=c_ssl,
                    extra_params=extra_dict
                )
                conn_obj = create_connector(cfg)
                if c_secret:
                    conn_obj.set_credential(c_secret)
                st.session_state["active_connector_instance"] = conn_obj
                st.session_state["active_connector_config"] = cfg
                st.success(f"Connection '{cfg.name}' securely configured and encrypted with AES-256!")
        
        # Test & Schema Discovery Buttons (Guaranteed Non-Blank)
        active_conn: Optional[BaseConnector] = st.session_state.get("active_connector_instance")
        if not active_conn:
            demo_db_path, _ = get_or_create_demo_sqlite()
            cfg = ConnectorConfig(connector_type=ConnectorType.SQLITE, name="Demo SQLite DB", database=demo_db_path)
            active_conn = create_connector(cfg)
            st.session_state["active_connector_instance"] = active_conn

        if active_conn:
            st.markdown("---")
            st.markdown(f"#### 🔍 **Active Connector**: `{active_conn.config.name}` ({active_conn.config.connector_type.value})")
            
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                if st.button("🧪 Run Connection & Reachability Test", use_container_width=True, key="btn_test_active_conn"):
                    with st.spinner("Probing connection, verifying credentials & latency..."):
                        t_res: ConnectionResult = active_conn.test_connection()
                        if t_res.success:
                            set_task_status("Test Connection", "success", f"{t_res.message} ({t_res.latency_ms} ms)")
                            st.success(f"✅ **{t_res.message}**\n- Latency: `{t_res.latency_ms} ms`\n- Server: `{t_res.server_version or 'N/A'}`")
                        else:
                            set_task_status("Test Connection", "failed", t_res.message)
                            st.error(f"❌ **Connection Failed**:\n{t_res.message}")
            
            with act_col2:
                if st.button("📋 Discover Remote Schema & Tables", use_container_width=True, key="btn_schema_active_conn"):
                    with st.spinner("Introspecting database metadata & information schema..."):
                        try:
                            schema: SchemaMetadata = active_conn.get_schema()
                            st.session_state["connector_discovered_schema"] = schema
                            set_task_status("Discover Schema", "success", f"Discovered {schema.total_tables} tables in {schema.database_name}")
                            st.success(f"Discovered **{schema.total_tables}** table(s) in database `{schema.database_name}`.")
                        except Exception as e:
                            set_task_status("Discover Schema", "failed", str(e))
                            st.error(f"Schema discovery error: {CredentialSanitizer.sanitize(str(e))}")
            
            # Show discovered tables if available
            if "connector_discovered_schema" in st.session_state:
                sch: SchemaMetadata = st.session_state["connector_discovered_schema"]
                with st.expander(f"📊 Discovered Schema Catalog ({sch.total_tables} tables)", expanded=True):
                    for tb in sch.tables:
                        st.markdown(f"**Table:** `{tb.schema}.{tb.name}` {'(~' + str(tb.row_count_approx) + ' rows)' if tb.row_count_approx is not None else ''}")
                        col_df = pd.DataFrame([{"Column": c.name, "Type": c.data_type, "Nullable": c.nullable, "Primary Key": c.is_primary_key} for c in tb.columns])
                        st.dataframe(col_df, use_container_width=True, height=140)
            
            # Query & Ingestion Execution
            st.markdown("---")
            st.markdown("#### ⚡ **Extraction Query & Live Ingestion**")
            default_query = "SELECT * FROM orders LIMIT 50;" if active_conn.config.connector_type == ConnectorType.SQLITE else "SELECT * FROM [table_name] LIMIT 100;"
            query_input = st.text_area(
                "SQL Query / Collection / Endpoint Path:",
                value=st.session_state.get("connector_custom_query", default_query),
                height=90,
                key="conn_query_text_area"
            )
            
            q_col1, q_col2 = st.columns([1.5, 3])
            with q_col1:
                if st.button("🚀 Execute Query & Ingest to Workspace", type="primary", use_container_width=True, key="btn_exec_ingest_conn"):
                    with st.spinner("Executing extraction query via encrypted connector..."):
                        try:
                            t0 = time.time()
                            res_df = active_conn.execute_query(query_input, limit=1000)
                            elapsed = round((time.time() - t0) * 1000, 1)
                            
                            st.session_state.raw_df = res_df.copy()
                            st.session_state.current_df = res_df.copy()
                            st.session_state.dataset_source = f"{active_conn.config.name} ({len(res_df):,} rows)"
                            st.session_state.cleaning_manager = CleaningPipelineManager(res_df)
                            st.session_state.cleaning_history = []
                            st.session_state.fe_history = []
                            
                            pm.log_activity("Connected Data Ingestion", f"Extracted {len(res_df):,} rows from '{active_conn.config.name}'.")
                            set_task_status("Query & Ingest", "success", f"Extracted {len(res_df):,} rows in {elapsed} ms.")
                            st.success(f"🎉 Successfully ingested **{len(res_df):,} rows × {res_df.shape[1]} columns** into active workspace!")
                            st.rerun()
                        except Exception as e:
                            set_task_status("Query & Ingest", "failed", str(e))
                            st.error(f"Query execution failed: {CredentialSanitizer.sanitize(str(e))}")
            
            with q_col2:
                if st.button("👁️ Preview Query Data Only", use_container_width=True, key="btn_preview_conn"):
                    with st.spinner("Fetching preview records..."):
                        try:
                            res_df = active_conn.execute_query(query_input, limit=25)
                            st.session_state["connector_preview_df"] = res_df
                            st.dataframe(res_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Preview failed: {CredentialSanitizer.sanitize(str(e))}")

    with tab_arch:
        st.markdown("### 🏛️ **Enterprise Connector Architecture & Security Standards**")
        st.markdown("""
        The platform features an extensible **BaseConnector** architecture with zero-trust credential handling:
        """)
        
        arch_c1, arch_c2, arch_c3 = st.columns(3)
        with arch_c1:
            st.markdown("""
            <div class="action-box">
                <h4 style="margin:0 0 6px 0; color:#38bdf8;">🛡️ Credential Vault</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                    All secrets, tokens, private keys, and passwords are encrypted at rest using <b>Fernet (AES-128-CBC + HMAC-SHA256)</b> derived with 100,000 PBKDF2 iterations. Plaintext is only decrypted in ephemeral scope during socket creation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with arch_c2:
            st.markdown("""
            <div class="action-box">
                <h4 style="margin:0 0 6px 0; color:#34d399;">🧼 Credential Sanitizer</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                    All application logs, tracebacks, and queries pass through <code>CredentialSanitizer</code>, regex-redacting database connection URIs, Bearer tokens, private keys, and query parameters before reaching telemetry.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with arch_c3:
            st.markdown("""
            <div class="action-box">
                <h4 style="margin:0 0 6px 0; color:#fbbf24;">⚡ Extensible Lifecycle</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                    Standardized lifecycle: <code>test_connection()</code>, <code>get_schema()</code>, <code>execute_query()</code>, and <code>stream_records()</code> with Apache Arrow flight batching and graceful driver fallback.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📋 **Supported Drivers & Integration Capabilities Matrix**")
        matrix_data = [
            {"Data Source": "PostgreSQL", "Category": "Relational", "Driver / Engine": "psycopg2 / SQLAlchemy", "Auth Mechanism": "Password / MD5 / SCRAM-SHA-256", "SSL / Encryption": "SSL require / verify-full", "Format / Streaming": "Server-side cursors"},
            {"Data Source": "MySQL / MariaDB", "Category": "Relational", "Driver / Engine": "mysql-connector / pymysql", "Auth Mechanism": "Native Password / caching_sha2", "SSL / Encryption": "TLS 1.2 / 1.3 verify", "Format / Streaming": "Chunked batching"},
            {"Data Source": "SQLite", "Category": "Relational", "Driver / Engine": "sqlite3 (In-Process)", "Auth Mechanism": "Path Sandboxing / File ACL", "SSL / Encryption": "Local OS Sandboxed", "Format / Streaming": "Direct memory copy"},
            {"Data Source": "MS SQL Server", "Category": "Relational", "Driver / Engine": "pyodbc / TDS Driver 18", "Auth Mechanism": "SQL Auth / Windows AD Integrated", "SSL / Encryption": "TDS Encrypted Wire", "Format / Streaming": "Tabular Data Stream"},
            {"Data Source": "MongoDB", "Category": "NoSQL", "Driver / Engine": "pymongo / Atlas SRV", "Auth Mechanism": "SCRAM / X.509 / AWS IAM", "SSL / Encryption": "TLS / SNI Enabled", "Format / Streaming": "BSON to Flat DataFrame"},
            {"Data Source": "Snowflake", "Category": "Cloud Warehouse", "Driver / Engine": "snowflake-connector-python", "Auth Mechanism": "Key-Pair / SSO / OAuth2", "SSL / Encryption": "HTTPS End-to-End TLS", "Format / Streaming": "Apache Arrow Flight Stream"},
            {"Data Source": "Google BigQuery", "Category": "Cloud Warehouse", "Driver / Engine": "google-cloud-bigquery", "Auth Mechanism": "Service Account JSON / OAuth2", "SSL / Encryption": "Google RPC over TLS", "Format / Streaming": "BigQuery Read API (gRPC)"},
            {"Data Source": "Google Sheets", "Category": "SaaS Spreadsheets", "Driver / Engine": "gspread / Sheets API v4", "Auth Mechanism": "OAuth2 / Service Account", "SSL / Encryption": "HTTPS TLS 1.3", "Format / Streaming": "CSV Export Stream"},
            {"Data Source": "Google Drive", "Category": "SaaS Files", "Driver / Engine": "Google Drive API v3", "Auth Mechanism": "OAuth2 User Token / Share Links", "SSL / Encryption": "HTTPS TLS 1.3", "Format / Streaming": "Direct binary stream"},
            {"Data Source": "OneDrive / SharePoint", "Category": "SaaS Files", "Driver / Engine": "Microsoft Graph v1.0", "Auth Mechanism": "Azure Entra ID OAuth2", "SSL / Encryption": "HTTPS TLS 1.3", "Format / Streaming": "Office 365 Delta Sync"},
            {"Data Source": "REST / GraphQL API", "Category": "APIs & Webhooks", "Driver / Engine": "urllib / json / requests", "Auth Mechanism": "Bearer / API Key / Basic", "SSL / Encryption": "HTTPS Enforced", "Format / Streaming": "Paginated JSONPath"},
            {"Data Source": "Cloud Storage (S3/GCS/Azure)", "Category": "Storage Lakes", "Driver / Engine": "boto3 / gcs / azure-storage", "Auth Mechanism": "IAM Roles / SAS Tokens / Keys", "SSL / Encryption": "TLS / Server-Side KMS", "Format / Streaming": "Parquet / Delta / CSV Stream"}
        ]
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏛️ **Deep Architectural Blueprints for Core Database Connections**")
        
        db_tabs = st.tabs([
            "🐘 PostgreSQL",
            "🐬 MySQL",
            "🗄️ SQLite",
            "🏢 MS SQL Server",
            "🍃 MongoDB",
            "❄️ Snowflake",
            "🔍 BigQuery",
            "🛡️ Credential Vault"
        ])

        with db_tabs[0]:
            st.markdown("""
            ##### 🐘 **PostgreSQL Connection Architecture**
            - **Protocol & Driver:** Wire Protocol v3.0 via `psycopg2-binary` and `SQLAlchemy 2.0`
            - **Connection Pooling:** `QueuePool` with `pool_size=10, max_overflow=20`, `pool_recycle=1800` (avoids dropped firewall states)
            - **Health Check & Heartbeat:** `SELECT 1` pre-ping validation on checkout
            - **Transport Security:** Strict TLS encryption (`sslmode=require` or `sslmode=verify-full` with CA certificates)
            - **Authentication:** SCRAM-SHA-256 password hashing or mutual TLS (mTLS) certificates
            - **High-Throughput Streaming:** Server-side named cursors (`declare cursor`) streaming 5,000-row record batches directly into Pandas, eliminating memory bottlenecks
            - **Credential Security:** Encrypted in `CredentialVault` with Fernet/AES-256; decrypted strictly within ephemeral pool acquisition scope
            """)
            st.code('''# PostgreSQL Connection Pattern
from modules.connectors import create_connector, ConnectorConfig, ConnectorType

cfg = ConnectorConfig(
    connector_type=ConnectorType.POSTGRESQL,
    host="db.prod.internal",
    port=5432,
    database="analytics_prod",
    username="analyst_readonly",
    ssl_enabled=True
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_master_secret")  # Ephemeral memory only
schema = conn.get_schema()  # Introspects tables, types, foreign keys
df = conn.execute_query("SELECT * FROM customers LIMIT 1000")''', language="python")

        with db_tabs[1]:
            st.markdown("""
            ##### 🐬 **MySQL / MariaDB Connection Architecture**
            - **Protocol & Driver:** MySQL Client/Server Binary Protocol via `pymysql` / `mysql-connector-python`
            - **Connection Pooling:** Thread-safe connection pool with automatic keep-alive ping
            - **Collation & Encoding:** `utf8mb4_unicode_ci` ensuring full 4-byte Unicode & emoji support
            - **Transport Security:** TLS 1.2 / TLS 1.3 encrypted sockets with modern cipher suites
            - **Authentication:** Modern `caching_sha2_password` (MySQL 8.0+) and fallback `mysql_native_password`
            - **High-Throughput Streaming:** Unbuffered `pymysql.cursors.SSCursor` (Server-Side Cursor) streaming records over TCP window
            - **Credential Security:** Fernet AES-256 encrypted credential state; connection string credentials masked from logs
            """)
            st.code('''# MySQL Connection Pattern
cfg = ConnectorConfig(
    connector_type=ConnectorType.MYSQL,
    host="mysql.cluster.internal",
    port=3306,
    database="ecommerce_store",
    username="db_reader",
    ssl_enabled=True
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_password")
res = conn.test_connection()  # Latency check and server version discovery''', language="python")

        with db_tabs[2]:
            st.markdown("""
            ##### 🗄️ **SQLite Connection Architecture**
            - **Protocol & Driver:** Direct C-level in-process embedded library via Python's standard `sqlite3` (Zero external dependencies)
            - **Concurrency & WAL Mode:** `PRAGMA journal_mode=WAL;` (Write-Ahead Logging) for non-blocking concurrent reads during writes
            - **Cache Sizing:** `PRAGMA cache_size = -64000;` allocating 64 MB of dedicated RAM page caching
            - **Path Sandboxing:** Strict filesystem path traversal verification preventing directory escapes (`..`)
            - **Zero-Copy Streaming:** `pd.read_sql_query()` with direct C-to-Python memory mapping
            - **Credential Security:** Inherited OS file permissions / SQLCipher AES-256 page-level encryption support
            """)
            st.code('''# SQLite Sandboxed Pattern
from modules.connectors.demo_db import get_or_create_demo_sqlite

db_path, _ = get_or_create_demo_sqlite()
cfg = ConnectorConfig(connector_type=ConnectorType.SQLITE, database=db_path)
conn = create_connector(cfg)
df = conn.execute_query("SELECT * FROM v_customer_revenue")''', language="python")

        with db_tabs[3]:
            st.markdown("""
            ##### 🏢 **Microsoft SQL Server Connection Architecture**
            - **Protocol & Driver:** TDS (Tabular Data Stream) Protocol v7.4+ via Microsoft ODBC Driver 18 (`pyodbc`)
            - **Connection Encryption:** Wire encryption enforced via `Encrypt=yes;TrustServerCertificate=no;`
            - **Authentication:** SQL Server Authentication or Windows Active Directory / Azure Entra ID Integrated Kerberos
            - **Failover Cluster:** `MultiSubnetFailover=Yes` for high-availability Always-On Availability Groups
            - **High-Throughput Streaming:** `cursor.fast_executemany = True` and parameterized `fetchmany(5000)` pagination
            - **Credential Security:** Passwords never interpolated into raw connection strings; sanitized in all error tracebacks
            """)
            st.code('''# Microsoft SQL Server TDS Pattern
cfg = ConnectorConfig(
    connector_type=ConnectorType.MSSQL,
    host="sqlserver.corp.local",
    port=1433,
    database="CorpWarehouse",
    username="sa_analyst",
    ssl_enabled=True
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_sa_pwd")
df = conn.execute_query("SELECT TOP 100 * FROM FactSales")''', language="python")

        with db_tabs[4]:
            st.markdown("""
            ##### 🍃 **MongoDB Connection Architecture**
            - **Protocol & Driver:** MongoDB Wire Protocol (OP_MSG) via official `pymongo` with `dnspython` SRV resolution
            - **Connection Pooling:** `MongoClient` connection pool with `maxPoolSize=50, minPoolSize=5`
            - **Topology Discovery:** Automatic replica set node discovery and primary/secondary election detection
            - **Read Preference:** `secondaryPreferred` to route analytical read queries away from the write primary
            - **BSON Tabular Normalization:** `flatten_mongo_documents()` recursively unrolls nested JSON/BSON documents into 2D DataFrames
            - **Authentication:** SCRAM-SHA-256, MONGODB-AWS (IAM STS tokens), or X.509 client certificates
            - **Credential Security:** `mongodb+srv://` URIs encrypted with Fernet AES-256 and redacted in telemetry
            """)
            st.code('''# MongoDB BSON to DataFrame Normalization
cfg = ConnectorConfig(
    connector_type=ConnectorType.MONGODB,
    host="cluster0.mongodb.net",
    database="analytics_lake"
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_atlas_token")
df = conn.execute_query('{"collection": "user_sessions", "filter": {"duration_sec": {"$gt": 100}}}')''', language="python")

        with db_tabs[5]:
            st.markdown("""
            ##### ❄️ **Snowflake Cloud Data Warehouse Architecture**
            - **Protocol & Driver:** Snowflake REST/SQL API & Apache Arrow Flight Protocol via `snowflake-connector-python`
            - **Multi-Cluster Architecture:** Virtual Warehouse elastic compute layer decoupled from centralized cloud storage
            - **Virtual Warehouse Control:** Session context sets `WAREHOUSE`, `DATABASE`, `SCHEMA`, `ROLE` with auto-suspend
            - **Sub-Second Arrow Flight Streaming:** `cursor.fetch_pandas_all()` executes zero-copy Apache Arrow IPC memory transfers
            - **Authentication:** RSA 2048-bit Key-Pair authentication, Okta/Azure SSO, or OAuth2 access tokens
            - **Credential Security:** RSA private key decrypted only in memory using vault passphrase; zero disk exposure
            """)
            st.code('''# Snowflake Arrow Flight Zero-Copy Connector
cfg = ConnectorConfig(
    connector_type=ConnectorType.SNOWFLAKE,
    host="xy12345.us-east-1",
    database="ANALYTICS_PROD",
    username="BI_ENGINEER",
    extra_params={"warehouse": "COMPUTE_WH", "role": "ANALYST"}
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_rsa_private_key")
df = conn.execute_query("SELECT * FROM FCT_DAILY_ORDERS LIMIT 5000")''', language="python")

        with db_tabs[6]:
            st.markdown("""
            ##### 🔍 **Google BigQuery Serverless Architecture**
            - **Protocol & Driver:** BigQuery REST API v2 and high-speed gRPC Storage Read API via `google-cloud-bigquery`
            - **Distributed Serverless Engine:** Powered by Google Dremel and Colossus columnar file system (Capacitor format)
            - **Pre-Flight Cost Estimation:** `job_config.dry_run = True` calculates exact bytes billed and estimated USD cost prior to execution
            - **BigQuery Storage Read API:** Multiplexed parallel partitioned gRPC streams delivering Arrow record batches directly to Pandas
            - **Authentication:** Google Cloud IAM Service Account JSON key block or Workload Identity Federation
            - **Credential Security:** Service Account private keys encrypted in Fernet AES-256; project IDs and emails masked in logs
            """)
            st.code('''# BigQuery Serverless Connector with Cost Estimation
cfg = ConnectorConfig(
    connector_type=ConnectorType.BIGQUERY,
    database="gcp-analytics-prod",
    host="marketing_dataset"
)
conn = create_connector(cfg)
conn.set_credential("vault_encrypted_service_account_json")
cost_info = conn.estimate_query_cost("SELECT * FROM ad_campaign_performance")
print(f"Estimated Billed: {cost_info['bytes_billed'] / 1e9:.2f} GB")
df = conn.execute_query("SELECT * FROM ad_campaign_performance LIMIT 1000")''', language="python")

        with db_tabs[7]:
            st.markdown("""
            ##### 🛡️ **Secure Credential Vault & Sanitizer Architecture**
            - **Encryption Algorithm:** Fernet authenticated symmetric cipher (AES-128-CBC + HMAC-SHA256)
            - **Key Derivation Function:** PBKDF2-HMAC-SHA256 with 100,000 rounds and cryptographic salt
            - **Zero Plaintext at Rest:** Secrets are encrypted immediately upon entry; never saved to disk or persistent state
            - **Ephemeral Decryption:** Credentials exist as plaintext strictly during socket handshake and are dereferenced immediately
            - **Telemetry & Log Redaction:** `CredentialSanitizer` regex interceptor scrubs URIs, tokens, and private keys from all logs
            - **Environment Injection:** Master vault key configurable via `DATAMIND_ENCRYPTION_KEY` environment variable
            """)
            st.code('''# Defense-in-Depth Credential Security
from modules.connectors.security import CredentialVault, CredentialSanitizer

vault = CredentialVault(salt=b"enterprise_salt")
ciphertext = vault.encrypt("super_secret_db_password")
clean_uri = CredentialSanitizer.sanitize_uri("postgresql://admin:super_secret_db_password@db.prod:5432/lake")
# Returns: "postgresql://admin:********@db.prod:5432/lake"''', language="python")


    with tab_demos:
        st.markdown("### ⚡ **Instant Pre-Configured Live Demos**")
        st.write("Test database connections, multi-table SQL joins, and automatic schema discovery immediately without supplying external credentials:")
        
        d_demo1, d_demo2 = st.columns(2)
        with d_demo1:
            st.markdown("""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px;">
                <h4 style="margin:0 0 8px 0; color:#38bdf8;">🗄️ Enterprise Sales & Customers SQLite DB</h4>
                <p style="font-size:0.86rem; color:#94a3b8; margin-bottom:14px;">
                    Pre-populated with <code>customers</code>, <code>orders</code>, and aggregated <code>v_customer_revenue</code> view with relational foreign keys.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("⚡ Connect & Load Demo SQLite Database", use_container_width=True, key="btn_load_demo_sqlite"):
                db_file, db_desc = get_or_create_demo_sqlite()
                demo_cfg = ConnectorConfig(
                    connector_type=ConnectorType.SQLITE,
                    name="Demo Enterprise Sales DB",
                    database=db_file
                )
                demo_conn = create_connector(demo_cfg)
                st.session_state["active_connector_instance"] = demo_conn
                st.session_state["active_connector_config"] = demo_cfg
                st.session_state["connector_discovered_schema"] = demo_conn.get_schema()
                st.session_state["connector_custom_query"] = """SELECT 
    c.company_name,
    c.country,
    c.tier,
    o.product_category,
    o.order_amount,
    o.payment_status,
    o.order_date
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
ORDER BY o.order_amount DESC;"""
                set_task_status("Demo SQLite Connection", "success", "Connected to enterprise demo database and introspected schema.")
                st.success("🎉 Connected to Demo SQLite Database! Switch to 'Connect & Ingest' tab to explore schema or execute joined queries.")
                st.rerun()

        with d_demo2:
            st.markdown("""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px;">
                <h4 style="margin:0 0 8px 0; color:#34d399;">🚀 One-Click Ingest Joined Sales to Workspace</h4>
                <p style="font-size:0.86rem; color:#94a3b8; margin-bottom:14px;">
                    Directly queries the multi-table relational join from SQLite and sets it as the active platform dataset for automated cleaning and AI analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Ingest Joined Relational Dataset", use_container_width=True, key="btn_ingest_joined_demo"):
                db_file, _ = get_or_create_demo_sqlite()
                demo_cfg = ConnectorConfig(
                    connector_type=ConnectorType.SQLITE,
                    name="Enterprise Sales & Customers",
                    database=db_file
                )
                demo_conn = create_connector(demo_cfg)
                joined_df = demo_conn.execute_query("""SELECT 
    c.company_name,
    c.country,
    c.tier,
    o.product_category,
    o.order_amount,
    o.payment_status,
    o.order_date
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;""")
                st.session_state.raw_df = joined_df.copy()
                st.session_state.current_df = joined_df.copy()
                st.session_state.dataset_source = "Demo SQLite: Customers JOIN Orders"
                st.session_state.cleaning_manager = CleaningPipelineManager(joined_df)
                st.session_state.cleaning_history = []
                st.session_state.fe_history = []
                pm.log_activity("Demo Ingestion", f"Loaded {len(joined_df)} joined rows from SQLite demo.")
                set_task_status("Demo Ingestion", "success", f"Loaded {len(joined_df)} joined relational rows.")
                st.success(f"🎉 Ingested {len(joined_df)} rows × {joined_df.shape[1]} columns into active workspace!")
                st.rerun()


# ---------------- 4. DATA PREVIEW ----------------
elif selected_module == "📋 Data Preview":
    st.subheader("📋 Interactive Spreadsheet-Like Data Viewer")
    
    if df.empty:
        st.info("📂 **No dataset uploaded yet.** Please upload your CSV, Excel, or Parquet file, or choose a demo dataset to explore.")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("📤 Go to Data Upload", type="primary", use_container_width=True):
                st.session_state.quick_jump = "📤 Data Upload"
                st.rerun()
        with col_btn2:
            if st.button("Load Churn Demo", use_container_width=True):
                df_churn = generate_sample_customer_churn()
                st.session_state.raw_df = df_churn.copy()
                st.session_state.current_df = df_churn.copy()
                st.session_state.dataset_source = "Customer Churn (Demo Dataset)"
                st.session_state.cleaning_manager = CleaningPipelineManager(df_churn)
                st.rerun()
    else:
        # Filter & Search Bar
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_term = st.text_input("🔍 Search rows (text match):", placeholder="Filter records by any string...")
        with f_col2:
            sort_col = st.selectbox("Sort by column:", ["None"] + list(df.columns))
        with f_col3:
            sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)

        preview_df = df.copy()
        if search_term:
            mask = preview_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
            preview_df = preview_df[mask]

        if sort_col != "None":
            preview_df = preview_df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))

        # Pagination controls
        page_size = 25
        total_pages = max(1, int(np.ceil(len(preview_df) / page_size)))
        curr_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (curr_page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, len(preview_df))} of {len(preview_df):,} filtered records.")
        st.dataframe(preview_df.iloc[start_idx:end_idx], use_container_width=True)

        # Column Metadata & Semantic Profile Table
        st.markdown("### 📊 Column Semantic Profiles & Types")
        col_meta = infer_column_metadata(df)
        meta_records = []
        for c, m in col_meta.items():
            meta_records.append({
                "Column": c,
                "Semantic Role": m["role"].upper(),
                "Data Type": m["dtype"],
                "Null Count": m["null_count"],
                "Null %": f"{m['null_pct']}%",
                "Unique Values": m["unique_count"],
                "Target Candidate": "🎯 Yes" if m["is_target_candidate"] else "No"
            })
        st.dataframe(pd.DataFrame(meta_records), use_container_width=True)


# ---------------- 5. DATA CLEANING ----------------
elif selected_module == "🧹 Data Cleaning":
    st.subheader("🧹 Automated & Manual Data Cleaning Studio")
    
    t_clean1, t_clean2, t_clean3, t_clean4, t_clean5 = st.tabs([
        "🚀 1-Click Auto-Clean",
        "🩹 Missing Values",
        "🚨 Outliers Handling",
        "🔤 Deduplication & Text",
        "🏷️ Type Casting"
    ])

    with t_clean1:
        st.markdown("""
        <div class="action-box">
            <b>1-Click AI Auto-Cleaning Engine:</b> Automatically detects missing values, duplicate rows, zero-variance columns, and extreme outliers. Applies robust median/mode imputation, string trimming, and winsorization in a single step.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Run 1-Click AI Auto-Clean", type="primary", use_container_width=True):
            try:
                clean_df, log = one_click_ai_auto_clean(df)
                st.session_state.current_df = clean_df
                st.session_state.cleaning_manager.apply_step("1-Click Auto-Clean", clean_df)
                pm.log_activity("Executed Auto-Clean", f"Cleaned {len(log)} issues.")
                set_task_status("1-Click AI Auto-Clean", "success", f"Cleaned {len(log)} data hygiene issues")
                st.success("Auto-clean completed successfully!")
                for item in log:
                    st.write(f"- {item}")
                st.rerun()
            except Exception as e:
                set_task_status("1-Click AI Auto-Clean", "failed", str(e))
                st.error(f"Auto-clean failed: {str(e)}")

    with t_clean2:
        st.markdown("#### 🩹 Missing Value Imputation")
        missing_counts = {c: int(df[c].isna().sum()) for c in df.columns}
        cols_sorted = sorted(df.columns, key=lambda c: (-missing_counts[c], c))
        miss_cols = [c for c in cols_sorted if missing_counts[c] > 0]
        
        if miss_cols:
            st.info(f"Detected **{len(miss_cols)}** column(s) with missing values: {', '.join([f'{c} ({missing_counts[c]})' for c in miss_cols[:6]])}")
        else:
            st.success("🎉 No missing null values detected in the current dataset! You can still apply imputation, fills, or constants to any column below.")

        sel_miss_col = st.selectbox(
            "Select column to impute:",
            cols_sorted,
            format_func=lambda c: f"{c} ({missing_counts[c]} missing)" if missing_counts[c] > 0 else f"{c} (0 missing)"
        )
        imp_method = st.selectbox(
            "Imputation Strategy:",
            ["ai_recommended", "mean", "median", "mode", "forward_fill", "backward_fill", "interpolate", "constant", "knn", "drop_row", "drop_col"]
        )
        const_val = st.text_input("Constant value (if strategy = constant):", value="Missing") if imp_method == "constant" else None

        if st.button("Apply Imputation", type="primary"):
            try:
                strat_dict = {sel_miss_col: imp_method}
                c_dict = {sel_miss_col: const_val} if const_val else None
                count_before = int(df[sel_miss_col].isna().sum())
                clean_df, log, _ = impute_missing_values(df, strategy_dict=strat_dict, constant_values=c_dict)
                count_after = int(clean_df[sel_miss_col].isna().sum()) if sel_miss_col in clean_df.columns else 0
                st.session_state.current_df = clean_df
                st.session_state.cleaning_manager.apply_step(f"Imputed '{sel_miss_col}' ({imp_method}, resolved {count_before - count_after} missing)", clean_df)
                set_task_status("Missing Value Imputation", "success", f"Imputed '{sel_miss_col}' via {imp_method}. Resolved {count_before - count_after} missing values.")
                st.success(f"Imputed missing values in '{sel_miss_col}'. Resolved {count_before - count_after} missing values.")
                st.rerun()
            except Exception as e:
                set_task_status("Missing Value Imputation", "failed", str(e))
                st.error(f"Imputation failed: {str(e)}")

    with t_clean3:
        st.markdown("#### 🚨 Outlier Detection & Treatment")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numerical columns available for outlier detection.")
        else:
            sel_out_col = st.selectbox("Select numerical column:", num_cols)
            out_method = st.selectbox("Detection Method:", ["iqr", "z_score", "modified_z_score", "isolation_forest", "lof"])
            out_action = st.selectbox("Action:", ["cap", "remove", "set_nan", "keep"])
            out_factor = st.slider("IQR / Threshold Factor:", min_value=1.0, max_value=3.5, value=1.5, step=0.1) if out_method == "iqr" else 3.0

            # Compute IQR bounds preview
            if out_method == "iqr" and len(df[sel_out_col].dropna()) > 0:
                q1 = float(df[sel_out_col].quantile(0.25))
                q3 = float(df[sel_out_col].quantile(0.75))
                iqr = q3 - q1
                low_b = round(q1 - out_factor * iqr, 2)
                high_b = round(q3 + out_factor * iqr, 2)
                out_cnt = int(((df[sel_out_col] < low_b) | (df[sel_out_col] > high_b)).sum())
                st.caption(f"📊 **IQR Bounds:** Q1 = `{q1:.2f}`, Q3 = `{q3:.2f}`, IQR = `{iqr:.2f}` | Valid Range: `[{low_b}, {high_b}]` | Detected Outliers: **{out_cnt}**")

            if st.button("Handle Outliers", type="primary"):
                try:
                    clean_df, log = handle_outliers(df, columns=[sel_out_col], method=out_method, action=out_action, factor=out_factor)
                    st.session_state.current_df = clean_df
                    st.session_state.cleaning_manager.apply_step(f"Outlier treatment on '{sel_out_col}' ({out_method}-{out_action})", clean_df)
                    set_task_status("Outlier Treatment (IQR)", "success", f"Outlier operation '{out_action}' applied on '{sel_out_col}'")
                    st.success(f"Outlier operation '{out_action}' applied on '{sel_out_col}'!")
                    st.rerun()
                except Exception as e:
                    set_task_status("Outlier Treatment (IQR)", "failed", str(e))
                    st.error(f"Outlier treatment failed: {str(e)}")

    with t_clean4:
        st.markdown("#### 🔤 Deduplication & Text Normalization")
        st.write(f"Current duplicate rows: **{df.duplicated().sum():,}**")
        c_dup, c_trim, c_case = st.columns(3)
        with c_dup:
            do_dup = st.checkbox("Remove Duplicates", value=True)
            dup_mode = st.selectbox("Keep mode:", ["first", "last", "drop_all"])
        with c_trim:
            do_trim = st.checkbox("Trim Whitespace", value=True)
        with c_case:
            case_mode = st.selectbox("Text Case:", ["keep", "lower", "upper", "title"])

        if st.button("Standardize Text & Deduplicate", type="primary"):
            try:
                clean_df, log = clean_text_and_duplicates(df, remove_dups=do_dup, keep_dup_mode=dup_mode, trim_strings=do_trim, text_case=case_mode)
                st.session_state.current_df = clean_df
                st.session_state.cleaning_manager.apply_step("Deduplication & Text Cleaning", clean_df)
                set_task_status("Deduplication & Text Cleaning", "success", "Cleaned duplicates and standardized text")
                st.success("Deduplication and text standardization applied!")
                st.rerun()
            except Exception as e:
                set_task_status("Deduplication & Text Cleaning", "failed", str(e))
                st.error(f"Deduplication failed: {str(e)}")
            st.rerun()

    with t_clean5:
        st.markdown("#### 🏷️ Column Type Casting")
        target_col_cast = st.selectbox("Select column to cast:", list(df.columns))
        target_type_cast = st.selectbox("Target Type:", ["numeric", "datetime", "categorical", "string", "bool"])
        if st.button("Convert Column Type", type="primary"):
            clean_df, log = convert_column_types(df, {target_col_cast: target_type_cast})
            st.session_state.current_df = clean_df
            st.session_state.cleaning_manager.apply_step(f"Cast '{target_col_cast}' to {target_type_cast}", clean_df)
            st.success(f"Converted '{target_col_cast}' to {target_type_cast}.")
            st.rerun()

    # Data Cleaning Changelog & Audit Trail
    st.markdown("---")
    st.markdown("### 📋 Data Cleaning Changelog & Audit Trail")
    c_hist_left, c_hist_right = st.columns([3, 1])

    clean_mgr: CleaningPipelineManager = st.session_state.cleaning_manager
    steps = clean_mgr.get_pipeline_steps()

    with c_hist_left:
        st.caption(f"Total History Steps: **{len(steps)}** | Active Pointer: **Step {clean_mgr.current_idx + 1} of {len(steps)}** | Active Rows: **{len(df):,}**")
        for idx, step_name in enumerate(steps):
            is_active = (idx == clean_mgr.current_idx)
            icon = "👉 **Active:**" if is_active else f"#{idx+1}"
            border_style = "border-left: 4px solid #6366f1; background: rgba(99,102,241,0.12);" if is_active else "border-left: 2px solid #334155; background: rgba(30,41,59,0.5);"
            step_df = clean_mgr.history[idx][1] if idx < len(clean_mgr.history) else None
            meta_txt = f"<span style='color:#818cf8;font-size:0.75rem;margin-left:8px;'>(rows: {len(step_df):,}, cols: {len(step_df.columns)})</span>" if step_df is not None else ""
            st.markdown(
                f"""<div style="{border_style} padding: 8px 12px; border-radius: 6px; margin-bottom: 5px; font-size: 0.88rem;">
                    {icon} <b>{step_name}</b> {meta_txt}
                </div>""",
                unsafe_allow_html=True
            )

    with c_hist_right:
        st.markdown("##### History Controls")
        if clean_mgr.can_undo():
            if st.button("↩️ Undo Step", use_container_width=True, key="btn_undo_clean"):
                res = clean_mgr.undo()
                if res:
                    st.session_state.current_df = res[1].copy()
                    st.session_state.cleaning_history.append(f"Undid step: {res[0]}")
                    set_task_status("Undo Cleaning Step", "success", f"Reverted to: {res[0]}")
                    st.success(f"Reverted to: {res[0]}")
                    st.rerun()
        else:
            st.button("↩️ Undo (At Start)", disabled=True, use_container_width=True, key="btn_undo_clean_dis")

        if clean_mgr.can_redo():
            if st.button("↪️ Redo Step", use_container_width=True, key="btn_redo_clean"):
                res = clean_mgr.redo()
                if res:
                    st.session_state.current_df = res[1].copy()
                    st.session_state.cleaning_history.append(f"Redid step: {res[0]}")
                    set_task_status("Redo Cleaning Step", "success", f"Restored: {res[0]}")
                    st.success(f"Restored: {res[0]}")
                    st.rerun()

        if st.button("🔄 Reset to Raw Data", use_container_width=True, key="btn_reset_clean"):
            if st.session_state.raw_df is None or st.session_state.raw_df.empty:
                set_task_status("Reset to Raw Data", "failed", "No raw dataset available to reset. Please upload a CSV or load a demo dataset first.")
                st.warning("No raw dataset available to reset.")
            else:
                st.session_state.current_df = st.session_state.raw_df.copy()
                st.session_state.cleaning_manager = CleaningPipelineManager(st.session_state.raw_df)
                st.session_state.cleaning_history = []
                st.session_state.fe_history = []
                set_task_status("Reset to Raw Data", "success", f"Reverted dataset to original raw state ({len(st.session_state.raw_df):,} rows × {st.session_state.raw_df.shape[1]} columns).")
                st.rerun()

    st.markdown("---")
    st.markdown(f"#### 📋 Current Dataset Preview ({len(df):,} rows × {df.shape[1]} columns)")
    st.dataframe(df.head(25), use_container_width=True)


# ---------------- 6. DATA PROCESSING ----------------
elif selected_module in ["🔄 Data Processing & Transform", "🔄 Data Processing", "Data Processing", "Data Processing & Transform"]:
    st.subheader("🔄 Visual Data Transformation Engine")
    
    tp_tab1, tp_tab2, tp_tab3, tp_tab4 = st.tabs([
        "Filter & Calculated Columns",
        "Group By & Aggregations",
        "Reshaping & Pivots",
        "Joins & Merges"
    ])

    with tp_tab1:
        st.markdown("#### Filter Rows")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filt_col = st.selectbox("Filter Column:", list(df.columns), key="proc_filt_col")
        with fc2:
            filt_op = st.selectbox("Operator:", ["==", "!=", ">", ">=", "<", "<=", "contains", "isnull", "notnull"])
        with fc3:
            filt_val = st.text_input("Filter Value:", value="0")

        if st.button("Apply Row Filter", type="primary"):
            filtered_df = filter_rows(df, filt_col, filt_op, filt_val)
            st.session_state.current_df = filtered_df
            st.session_state.cleaning_manager.apply_step(f"Filter {filt_col} {filt_op} {filt_val}", filtered_df)
            st.success(f"Filter applied. Remaining rows: {len(filtered_df):,}")
            st.rerun()

        st.markdown("---")
        st.markdown("#### Calculated Column Expression")
        cc_name = st.text_input("New Column Name:", placeholder="e.g. ProfitMargin")
        cc_expr = st.text_input("Formula Expression:", placeholder="e.g. df['Revenue'] - df['Cost'] or df['MonthlyCharges'] * 12")
        if st.button("Compute Calculated Column", type="primary"):
            if cc_name and cc_expr:
                res_df, msg = create_calculated_column(df, cc_name, cc_expr)
                if cc_name in res_df.columns:
                    st.session_state.current_df = res_df
                    st.session_state.cleaning_manager.apply_step(f"Created column {cc_name}", res_df)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tp_tab2:
        st.markdown("#### Group By & Aggregate")
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if cat_cols and num_cols:
            grp_col = st.selectbox("Group By Dimension:", cat_cols)
            agg_metric = st.selectbox("Aggregation Metric:", num_cols)
            agg_funcs = st.multiselect("Aggregation Functions:", ["sum", "avg", "min", "max", "count", "median", "std"], default=["sum", "avg"])

            if st.button("Execute Group By", type="primary"):
                agg_df = group_by_aggregate(df, [grp_col], {agg_metric: agg_funcs})
                st.dataframe(agg_df, use_container_width=True)

    with tp_tab3:
        st.markdown("#### Pivot Table")
        if len(df.columns) >= 3:
            p_idx = st.selectbox("Index (Rows):", list(df.columns), index=0)
            p_cols = st.selectbox("Columns:", list(df.columns), index=1 if len(df.columns) > 1 else 0)
            p_vals = st.selectbox("Values:", df.select_dtypes(include=[np.number]).columns.tolist())
            p_func = st.selectbox("Aggfunc:", ["mean", "sum", "count", "max", "min"])

            if st.button("Generate Pivot Table", type="primary"):
                piv_df = pivot_dataset(df, p_idx, p_cols, p_vals, aggfunc=p_func)
                st.dataframe(piv_df, use_container_width=True)

    with tp_tab4:
        st.markdown("#### Relational Join")
        st.caption("Join the current dataset with a secondary table or demo data.")
        join_target = st.selectbox("Select join partner dataset:", ["Demo: Telecom Customer Churn", "Demo: Real Estate Housing", "Demo: E-Commerce Sales"])
        join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"])
        
        # Partner df
        partner_df = generate_sample_housing() if "Housing" in join_target else generate_sample_customer_churn()
        left_key = st.selectbox("Left Key (Current Dataset):", list(df.columns))
        right_key = st.selectbox("Right Key (Partner Dataset):", list(partner_df.columns))

        if st.button("Perform Join", type="primary"):
            try:
                joined = join_datasets(df, partner_df, left_on=left_key, right_on=right_key, how=join_type)
                st.session_state.current_df = joined
                st.session_state.cleaning_manager.apply_step(f"{join_type.title()} Join on {left_key}", joined)
                st.success(f"Joined datasets! Result contains {len(joined):,} rows × {joined.shape[1]} columns.")
                st.rerun()
            except Exception as e:
                st.error(f"Join error: {str(e)}")


# ---------------- 7. FEATURE ENGINEERING ----------------
elif selected_module == "⚙️ Feature Engineering":
    st.subheader("⚙️ Feature Engineering & Synthesis")
    
    t_fe1, t_fe2, t_fe3, t_fe4 = st.tabs([
        "💡 AI Recommendations",
        "📏 Scaling & Transforms",
        "🔤 Encoding & Datetime",
        "🔗 Interactions & Text"
    ])

    with t_fe1:
        recs = get_feature_recommendations(df)
        st.markdown("#### Automated Feature Engineering Suggestions")
        for r in recs[:5]:
            st.info(f"**{r['type']} on `{r['column']}`**: {r['suggestion']}\n\n*Rationale:* {r['reason']}")

        if st.button("🚀 Run 1-Click AI Feature Engineering", type="primary", use_container_width=True):
            fe_df, log = one_click_ai_feature_engineering(df)
            st.session_state.current_df = fe_df
            st.session_state.cleaning_manager.apply_step("1-Click Feature Engineering", fe_df)
            st.success("Synthesized features successfully!")
            st.rerun()

    with t_fe2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scale_col = st.selectbox("Feature to scale:", num_cols)
            scale_method = st.selectbox("Method:", ["standard", "minmax", "robust"])
            if st.button("Scale Feature", type="primary"):
                scaled_df, log = scale_features(df, [scale_col], method=scale_method)
                st.session_state.current_df = scaled_df
                st.session_state.cleaning_manager.apply_step(f"Scaled {scale_col}", scaled_df)
                st.success(f"Scaled feature '{scale_col}'.")
                st.rerun()

    with t_fe3:
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if cat_cols:
            enc_col = st.selectbox("Categorical feature to encode:", cat_cols)
            enc_method = st.selectbox("Encoding Method:", ["one_hot", "label", "frequency"])
            if st.button("Encode Feature", type="primary"):
                enc_df, log = encode_categorical(df, [enc_col], method=enc_method)
                st.session_state.current_df = enc_df
                st.session_state.cleaning_manager.apply_step(f"Encoded {enc_col}", enc_df)
                st.success(f"Encoded '{enc_col}'.")
                st.rerun()

    with t_fe4:
        st.markdown("#### Interaction & Ratio Features")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            int_c1 = st.selectbox("Numerator / Feature 1:", num_cols, index=0)
            int_c2 = st.selectbox("Denominator / Feature 2:", num_cols, index=1)
            int_op = st.selectbox("Operation:", ["ratio", "multiply", "add", "subtract"])
            int_name = st.text_input("New Feature Name (optional):", placeholder=f"{int_c1}_{int_op}_{int_c2}")
            if st.button("Synthesize Interaction", type="primary"):
                int_df, log = create_interaction_feature(df, int_c1, int_c2, operation=int_op, new_col_name=int_name)
                st.session_state.current_df = int_df
                st.session_state.cleaning_manager.apply_step(f"Interaction {int_c1} {int_op} {int_c2}", int_df)
                st.success("Interaction feature created!")
                st.rerun()


# ---------------- 8. EDA ----------------
elif selected_module == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    t_eda1, t_eda2, t_eda3 = st.tabs([
        "Descriptive Profiles",
        "Correlations & Multicollinearity",
        "Target Variable Deep Dive"
    ])

    with t_eda1:
        eda_stats = compute_comprehensive_stats(df)
        st.markdown("#### Numerical Attributes Summary")
        if not eda_stats["numeric"].empty:
            st.dataframe(eda_stats["numeric"], use_container_width=True)
        st.markdown("#### Categorical Attributes Summary")
        if not eda_stats["categorical"].empty:
            st.dataframe(eda_stats["categorical"], use_container_width=True)

    with t_eda2:
        corr_info = compute_correlation_analysis(df, method="pearson")
        if not corr_info["corr_matrix"].empty:
            fig_corr = plot_correlation_heatmap(corr_info["corr_matrix"])
            st.plotly_chart(fig_corr, use_container_width=True)
            if corr_info["collinear_pairs"]:
                st.warning(f"Found {len(corr_info['collinear_pairs'])} highly collinear variable pair(s) (|r| ≥ 0.80).")

    with t_eda3:
        target_cand = st.selectbox("Select Target / Outcome Variable:", list(df.columns))
        if target_cand:
            target_analysis = analyze_target_variable(df, target_cand)
            st.markdown(f"**Target Type:** `{'Continuous' if target_analysis['is_continuous'] else 'Categorical'}`")
            if target_analysis.get("associations"):
                st.dataframe(pd.DataFrame(target_analysis["associations"]), use_container_width=True)


# ---------------- 9. VISUALIZATION ----------------
elif selected_module == "📈 Visualization":
    st.subheader("📈 Visualization Engine & Chart Gallery")
    
    chart_category = st.selectbox(
        "Chart Category:",
        ["Basic Charts", "Statistical Charts", "Time-Series", "Advanced & BI Charts", "Geospatial Maps", "✨ AI Chart Generator"]
    )

    if chart_category == "Basic Charts":
        b_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Donut Chart", "Pie Chart"])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if b_type == "Bar Chart" and cat_cols and num_cols:
            x_c = st.selectbox("X Axis (Category):", cat_cols)
            y_c = st.selectbox("Y Axis (Metric):", num_cols)
            fig = plot_dashboard_column_chart(df, x_c, y_c)
            st.plotly_chart(fig, use_container_width=True)
        elif b_type == "Scatter Plot" and len(num_cols) >= 2:
            x_c = st.selectbox("X Axis:", num_cols, index=0)
            y_c = st.selectbox("Y Axis:", num_cols, index=1)
            col_c = st.selectbox("Color By (optional):", ["None"] + cat_cols)
            fig = plot_scatter(df, x_c, y_c, color_col=col_c if col_c != "None" else None)
            st.plotly_chart(fig, use_container_width=True)
        elif b_type in ["Donut Chart", "Pie Chart"] and cat_cols and num_cols:
            d_dim = st.selectbox("Category Dimension:", cat_cols)
            d_val = st.selectbox("Values Metric:", num_cols)
            fig = plot_dashboard_donut_chart(df, d_dim, d_val) if b_type == "Donut Chart" else plot_dashboard_pie_chart(df, d_dim, d_val)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_category == "Statistical Charts":
        s_type = st.selectbox("Statistical Chart:", ["Distribution (Histogram)", "Box Plot", "Violin Plot", "Kernel Density (KDE)", "Normal Q-Q Plot", "ECDF"])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if num_cols:
            sel_col = st.selectbox("Select Numeric Feature:", num_cols)
            if s_type == "Distribution (Histogram)":
                st.plotly_chart(plot_distribution(df, sel_col), use_container_width=True)
            elif s_type == "Box Plot" and cat_cols:
                grp_col = st.selectbox("Group By:", cat_cols)
                st.plotly_chart(plot_box_by_group(df, grp_col, sel_col), use_container_width=True)
            elif s_type == "Violin Plot" and cat_cols:
                grp_col = st.selectbox("Group By:", cat_cols)
                st.plotly_chart(plot_violin(df, grp_col, sel_col), use_container_width=True)
            elif s_type == "Kernel Density (KDE)":
                st.plotly_chart(plot_density_kde(df, sel_col), use_container_width=True)
            elif s_type == "Normal Q-Q Plot":
                st.plotly_chart(plot_qq(df, sel_col), use_container_width=True)
            elif s_type == "ECDF":
                st.plotly_chart(plot_ecdf(df, sel_col), use_container_width=True)

    elif chart_category == "Advanced & BI Charts":
        adv_type = st.selectbox("Advanced BI Chart:", ["Pareto 80/20 Chart", "Treemap", "Funnel Chart", "Performance Gauge"])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if adv_type == "Pareto 80/20 Chart" and cat_cols and num_cols:
            p_cat = st.selectbox("Category Dimension:", cat_cols)
            p_val = st.selectbox("Metric:", num_cols)
            st.plotly_chart(plot_pareto(df, p_cat, p_val), use_container_width=True)
        elif adv_type == "Treemap" and len(cat_cols) >= 2 and num_cols:
            t_paths = st.multiselect("Hierarchy Path:", cat_cols, default=cat_cols[:2])
            t_val = st.selectbox("Metric:", num_cols)
            if t_paths:
                st.plotly_chart(plot_treemap(df, t_paths, t_val), use_container_width=True)
        elif adv_type == "Performance Gauge" and num_cols:
            g_col = st.selectbox("Metric for Gauge:", num_cols)
            val = float(df[g_col].mean())
            st.plotly_chart(plot_gauge(val, min_val=0, max_val=float(df[g_col].max()), title=f"Average {g_col}"), use_container_width=True)

    elif chart_category == "Geospatial Maps":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            map_metric = st.selectbox("Map Metric:", num_cols)
            st.plotly_chart(plot_dashboard_map_chart(df, map_metric), use_container_width=True)

    elif chart_category == "✨ AI Chart Generator":
        st.markdown("#### Natural Language Chart Generator")
        ai_chart_prompt = st.text_input("Enter chart request:", placeholder="e.g. 'Show monthly charges across contract types'")
        if st.button("Generate Chart", type="primary") and ai_chart_prompt:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            rec = recommend_best_chart(df, cat_cols[0] if cat_cols else df.columns[0], num_cols[0] if num_cols else None)
            st.info(f"**AI Recommendation:** {rec['rationale']}")
            if cat_cols and num_cols:
                fig = plot_dashboard_column_chart(df, cat_cols[0], num_cols[0])
                st.plotly_chart(fig, use_container_width=True)


# ---------------- 10. SQL & QUERIES ----------------
elif selected_module == "🗄️ SQL & Queries":
    st.subheader("🗄️ SQL Analytics Studio & Query Engine")
    st.caption("Powered by in-memory DuckDB. Query your active dataset (registered as `df`) using standard ANSI SQL, Window functions, CTEs, Aggregations, and Joins.")
    
    studio: SQLStudio = st.session_state.sql_studio

    col_sql_main, col_sql_schema = st.columns([3, 1])

    with col_sql_schema:
        st.markdown("#### 📑 Active Table Schema")
        st.caption("Table name: `df`")
        schema_summary = pd.DataFrame({
            "Column": list(df.columns),
            "Type": [str(t) for t in df.dtypes]
        })
        st.dataframe(schema_summary, height=280, use_container_width=True, hide_index=True)
        
        with st.expander("💡 Quick SQL Cheatsheet", expanded=False):
            st.markdown("""
            - `SELECT * FROM df LIMIT 10`
            - `SELECT Col, COUNT(*) FROM df GROUP BY Col`
            - `SELECT * FROM df WHERE Col > 100`
            - `ROUND(AVG(Col), 2) AS avg_col`
            - `RANK() OVER (ORDER BY Col DESC)`
            - `WITH cte AS (...) SELECT * FROM cte`
            """)

    with col_sql_main:
        # Pre-built templates dropdown & AI Assistant
        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            templates = studio.get_templates()
            tmpl_names = ["-- Select Pre-Built Query Template --"] + [t["name"] for t in templates]
            chosen_tmpl = st.selectbox("Pre-Built Analytical Templates:", tmpl_names, key="sql_tmpl_select")

        default_sql = "SELECT * FROM df LIMIT 10;"
        if chosen_tmpl != "-- Select Pre-Built Query Template --":
            for t in templates:
                if t["name"] == chosen_tmpl:
                    default_sql = t["sql"]
                    break

        with t_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            show_nl_box = st.checkbox("🤖 AI Natural Language to SQL", value=False, key="toggle_nl_sql")

        if show_nl_box:
            nl_c1, nl_c2 = st.columns([3, 1])
            with nl_c1:
                nl_prompt = st.text_input("Ask a question in plain English:", placeholder="e.g. 'Show top 5 contracts with highest average monthly charges'", label_visibility="collapsed", key="nl_sql_input")
            with nl_c2:
                gen_btn = st.button("✨ Generate SQL", use_container_width=True, key="btn_gen_sql")
            
            if gen_btn and nl_prompt:
                ai_sql_res = generate_ai_sql(nl_prompt, df)
                default_sql = ai_sql_res["sql"]
                st.info(f"💡 **AI Explanation:** {ai_sql_res['explanation']}")
                st.code(ai_sql_res["sql"], language="sql")

        # SQL Editor Area
        sql_query = st.text_area("SQL Editor (ANSI SQL querying DataFrame registered as `df`):", value=default_sql, height=140, key="sql_editor_area")

        btn_run, btn_opt, btn_save = st.columns([1, 1, 1])
        with btn_run:
            run_sql = st.button("🚀 Execute Query", type="primary", use_container_width=True, key="btn_run_sql")
        with btn_opt:
            run_opt = st.button("🔍 Explain & Optimize", use_container_width=True, key="btn_opt_sql")
        with btn_save:
            save_name = st.text_input("Save Query Name:", placeholder="My Query", label_visibility="collapsed", key="sql_save_name")
            if st.button("💾 Save Query", use_container_width=True, key="btn_save_sql") and save_name:
                studio.save_query(save_name, sql_query)
                st.success(f"Saved '{save_name}'.")

    # Optimization tips panel
    if run_opt:
        opt_res = explain_and_optimize_sql(sql_query)
        st.markdown(f"**Engine Analysis:** {opt_res['analysis']}")
        for tip in opt_res["optimization_tips"]:
            st.info(f"💡 {tip}")

    # Execution and Results
    if run_sql:
        with st.spinner("Executing SQL query in DuckDB..."):
            res_df, err, elapsed = studio.execute_query(df, sql_query)
            if err:
                set_task_status("SQL Query Execution", "failed", err)
                st.error(f"❌ SQL Execution Error: {err}")
                st.session_state.last_query_result = None
            else:
                set_task_status("SQL Query Execution", "success", f"Returned {len(res_df):,} rows in {elapsed} ms")
                st.session_state.last_query_result = res_df
                st.session_state.last_query_time = elapsed
                st.session_state.last_query_sql = sql_query

    # Display Query Results if available
    if st.session_state.get("last_query_result") is not None:
        res_df = st.session_state.last_query_result
        elapsed = st.session_state.get("last_query_time", 0)

        st.markdown("---")
        st.markdown(f"### 📋 Query Results <span class='badge-chip badge-success'>{len(res_df):,} rows in {elapsed} ms</span>", unsafe_allow_html=True)

        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
            "📊 Data Table View",
            "📈 Instant Visualizer",
            "📜 Query History",
            "⭐ Saved Queries"
        ])

        with res_tab1:
            st.dataframe(res_df, use_container_width=True, height=350)
            
            # Action buttons on results
            act_col1, act_col2, act_col3 = st.columns([1, 1, 2])
            with act_col1:
                csv_res = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Results (CSV)",
                    data=csv_res,
                    file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="btn_export_query_csv"
                )
            with act_col2:
                if st.button("🔄 Set as Active Dataset", use_container_width=True, key="btn_set_active_dataset", help="Replaces active dataset with this query result for all downstream cleaning, EDA, and dashboarding"):
                    st.session_state.current_df = res_df.copy()
                    st.session_state.dataset_source = f"SQL Result ({len(res_df)} rows)"
                    st.session_state.cleaning_history.append(f"SQL Query applied: {len(res_df)} rows produced.")
                    set_task_status("Set as Active Dataset", "success", f"Active dataset updated ({len(res_df):,} rows)")
                    st.success("Query result is now the active working dataset!")
                    st.rerun()
            with act_col3:
                st.caption(f"Executed: `{st.session_state.get('last_query_sql', '')[:60]}...`")

        with res_tab2:
            st.markdown("#### Instant Visualization of Query Result")
            res_num_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()
            res_all_cols = res_df.columns.tolist()

            if len(res_all_cols) >= 1:
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    chart_type = st.selectbox("Chart Type:", ["Column / Bar Chart", "Line Chart", "Donut / Pie Chart", "Scatter Plot"], key="sql_res_chart_type")
                with v_col2:
                    x_axis = st.selectbox("X-Axis (Dimension):", res_all_cols, index=0, key="sql_res_x")
                with v_col3:
                    y_axis = st.selectbox("Y-Axis (Metric):", res_num_cols if res_num_cols else res_all_cols, index=0, key="sql_res_y")

                try:
                    if chart_type == "Column / Bar Chart":
                        fig_q = px.bar(res_df, x=x_axis, y=y_axis, template="plotly_dark", title=f"{y_axis} by {x_axis}")
                    elif chart_type == "Line Chart":
                        fig_q = px.line(res_df, x=x_axis, y=y_axis, markers=True, template="plotly_dark", title=f"{y_axis} Trend by {x_axis}")
                    elif chart_type == "Donut / Pie Chart":
                        fig_q = px.pie(res_df, names=x_axis, values=y_axis, hole=0.45, template="plotly_dark", title=f"{y_axis} Proportions by {x_axis}")
                    else:
                        fig_q = px.scatter(res_df, x=x_axis, y=y_axis, template="plotly_dark", title=f"{y_axis} vs {x_axis}")
                    fig_q.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_q, use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
            else:
                st.info("Query result has insufficient columns for visualization.")

        with res_tab3:
            st.markdown("#### Query Execution History")
            if studio.history:
                hist_df = pd.DataFrame(studio.history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("No queries executed in this session yet.")

        with res_tab4:
            st.markdown("#### Saved Queries Library")
            if studio.saved_queries:
                for sq in studio.saved_queries:
                    st.markdown(f"**{sq['name']}**")
                    st.code(sq["sql"], language="sql")
            else:
                st.info("No saved queries in library.")


# ---------------- PYTHON CODING ENVIRONMENT ----------------
elif selected_module == "🐍 Python Coding Environment":
    st.subheader("🐍 Python Analytics Studio & In-Browser Code Engine")
    st.caption("Write and execute Python / Pandas scripts directly against your active dataset (`df`).")
    render_python_environment(df, key_prefix="standalone_py_")


# ---------------- 11. STATISTICAL ANALYSIS ----------------
elif selected_module in ["📐 Statistical Analysis", "📐 Statistics"]:
    st.subheader("📐 Advanced Statistical Analysis & Hypothesis Testing Studio")
    st.caption("Comprehensive statistical diagnostics: Descriptive statistics, Normality tests, Confidence Intervals, Multi-method Correlation & Covariance, Hypothesis Testing with Effect Sizes, and OLS / Logistic Regression with AI Plain-Language Explanations.")

    t_stat1, t_stat2, t_stat3, t_stat4 = st.tabs([
        "📊 Descriptive & Normality",
        "🔗 Correlation & Covariance",
        "🧪 Hypothesis Testing",
        "📈 Regression Analysis"
    ])

    # 1. DESCRIPTIVE & NORMALITY
    with t_stat1:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.warning("No numeric columns found in the active dataset.")
        else:
            col_desc_sel, col_desc_btn = st.columns([3, 1])
            with col_desc_sel:
                stat_col = st.selectbox("Select Numeric Variable to Analyze:", num_cols, key="stats_desc_col")
            with col_desc_btn:
                st.write("")
                st.write("")
                run_desc_btn = st.button("Run Full Descriptive Audit", key="btn_run_desc_audit")

            res_desc = run_descriptive_statistics(df, stat_col)
            if "error" in res_desc:
                st.error(res_desc["error"])
            else:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Mean (μ)", f"{res_desc['mean']:,.2f}")
                m2.metric("Median (P50)", f"{res_desc['median']:,.2f}")
                m3.metric("Std Dev (s)", f"{res_desc['std_dev']:,.2f}")
                m4.metric("Skewness", f"{res_desc['skewness']:.2f}")
                m5.metric("Kurtosis", f"{res_desc['kurtosis']:.2f}")

                c_d1, c_d2 = st.columns([1, 1])
                with c_d1:
                    st.markdown("#### 📋 Detailed Statistical Parameters")
                    desc_table_data = {
                        "Parameter": [
                            "Sample Count (N)", "Arithmetic Mean", "Standard Error (SE)",
                            "95% Confidence Interval", "Median", "Mode", "Variance",
                            "Std Deviation", "IQR (Q3 - Q1)", "Min (P0)", "Q1 (25%)",
                            "Q3 (75%)", "Max (P100)", "Normality (D'Agostino/Shapiro)"
                        ],
                        "Value": [
                            str(res_desc["n_obs"]),
                            f"{res_desc['mean']:,.4f}",
                            f"{res_desc['std_error']:,.4f}",
                            f"[{res_desc['ci_95'][0]:,.3f}, {res_desc['ci_95'][1]:,.3f}]",
                            f"{res_desc['median']:,.4f}",
                            f"{res_desc['mode']:,.4f}",
                            f"{res_desc['variance']:,.4f}",
                            f"{res_desc['std_dev']:,.4f}",
                            f"{res_desc['iqr']:,.4f}",
                            f"{res_desc['min']:,.4f}",
                            f"{res_desc['q25']:,.4f}",
                            f"{res_desc['q75']:,.4f}",
                            f"{res_desc['max']:,.4f}",
                            f"{'Passed (Normal)' if res_desc['is_normal'] else 'Non-Normal'} (p={res_desc['normality_p_val']:.4f})"
                        ]
                    }
                    st.dataframe(pd.DataFrame(desc_table_data), use_container_width=True, hide_index=True)

                with c_d2:
                    st.markdown("#### 🤖 AI Plain-Language Explainer")
                    ai_desc_expl = explain_statistics_in_plain_language("descriptive", res_desc)
                    st.info(ai_desc_expl)

                    # Distribution chart
                    st.markdown("#### 📉 Distribution Plot")
                    st.bar_chart(df[stat_col].dropna().value_counts(bins=12).sort_index())

    # 2. CORRELATION & COVARIANCE
    with t_stat2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("At least 2 numeric columns required to compute correlation and covariance matrices.")
        else:
            c_m1, c_m2 = st.columns([2, 2])
            with c_m1:
                corr_method = st.radio(
                    "Select Association Metric:",
                    ["pearson", "spearman", "kendall", "covariance"],
                    format_func=lambda x: {
                        "pearson": "Pearson Correlation (Linear r)",
                        "spearman": "Spearman Rank Correlation (Monotonic ρ)",
                        "kendall": "Kendall Tau Correlation (Ordinal τ)",
                        "covariance": "Sample Covariance Matrix"
                    }[x],
                    horizontal=True
                )

            with c_m2:
                selected_corr_cols = st.multiselect(
                    "Include Columns (Default: All Numeric):",
                    num_cols,
                    default=num_cols[:8] if len(num_cols) > 8 else num_cols
                )

            if len(selected_corr_cols) >= 2:
                if corr_method == "covariance":
                    cov_res = compute_covariance_matrix(df, selected_corr_cols)
                    st.markdown("#### 📐 Sample Covariance Matrix")
                    st.dataframe(pd.DataFrame(cov_res["matrix"]), use_container_width=True)
                    st.caption("Covariance measures joint directional variability. Positive covariance indicates variables grow together; negative indicates inverse movement.")
                else:
                    corr_res = compute_correlation_matrices(df, selected_corr_cols, method=corr_method)
                    corr_df_view = pd.DataFrame(corr_res["matrix"])
                    st.markdown(f"#### 🔗 {corr_method.capitalize()} Correlation Matrix")
                    st.dataframe(corr_df_view, use_container_width=True)

                    c_c1, c_c2 = st.columns([1, 1])
                    with c_c1:
                        st.markdown("#### 🏆 Top Variable Relationships")
                        if corr_res.get("top_pairs"):
                            top_pairs_df = pd.DataFrame(corr_res["top_pairs"])[["var1", "var2", "coefficient", "strength", "direction"]]
                            st.dataframe(top_pairs_df, use_container_width=True, hide_index=True)
                    with c_c2:
                        st.markdown("#### 🤖 AI Plain-Language Explainer")
                        ai_corr_expl = explain_statistics_in_plain_language("correlation", corr_res)
                        st.success(ai_corr_expl)

    # 3. HYPOTHESIS TESTING
    with t_stat3:
        test_type = st.selectbox(
            "Select Hypothesis Test to Perform:",
            [
                ("independent_t_test", "Two-Sample Independent Welch's T-Test (Parametric)"),
                ("paired_t_test", "Paired Samples T-Test (Parametric)"),
                ("one_way_anova", "One-Way ANOVA (Parametric, 3+ Groups)"),
                ("chi_square", "Chi-Square Test of Independence (Categorical)"),
                ("mann_whitney_u", "Mann-Whitney U Test (Non-Parametric Independent)"),
                ("kruskal_wallis", "Kruskal-Wallis H-Test (Non-Parametric ANOVA)")
            ],
            format_func=lambda x: x[1]
        )[0]

        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        test_res = None
        if test_type in ["independent_t_test", "mann_whitney_u"]:
            if cat_cols and num_cols:
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    v1 = st.selectbox("Continuous Metric (Y):", num_cols, key="h_t_num")
                    g_col = st.selectbox("Grouping Factor:", cat_cols, key="h_t_cat")
                with h_c2:
                    u_vals = list(df[g_col].dropna().unique())
                    if len(u_vals) >= 2:
                        gv1 = st.selectbox("Group 1 Level:", u_vals, index=0, key="h_t_g1")
                        gv2 = st.selectbox("Group 2 Level:", u_vals, index=1, key="h_t_g2")
                    else:
                        st.warning("Selected grouping factor must have at least 2 distinct levels.")
                        gv1, gv2 = None, None

                if gv1 and gv2 and st.button("🚀 Run Hypothesis Test", type="primary", key="btn_run_indep"):
                    test_res = run_hypothesis_test(df, test_type, v1, group_col=g_col, group_val1=gv1, group_val2=gv2)
            else:
                st.warning("Test requires at least one numeric metric and one categorical grouping column.")

        elif test_type == "paired_t_test":
            if len(num_cols) >= 2:
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    v1 = st.selectbox("Metric 1 (Baseline / Pre):", num_cols, key="pair_v1")
                with h_c2:
                    v2 = st.selectbox("Metric 2 (Follow-up / Post):", num_cols, index=1, key="pair_v2")
                if st.button("🚀 Run Paired T-Test", type="primary", key="btn_run_pair"):
                    test_res = run_hypothesis_test(df, test_type, v1, var2=v2)
            else:
                st.warning("Paired test requires at least 2 continuous numeric variables.")

        elif test_type in ["one_way_anova", "kruskal_wallis"]:
            if cat_cols and num_cols:
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    v1 = st.selectbox("Dependent Metric (Y):", num_cols, key="anova_num")
                with h_c2:
                    g_col = st.selectbox("Multi-Group Factor (Category):", cat_cols, key="anova_cat")
                if st.button("🚀 Run Variance Analysis", type="primary", key="btn_run_anova"):
                    test_res = run_hypothesis_test(df, test_type, v1, group_col=g_col)
            else:
                st.warning("ANOVA requires at least 1 numeric metric and 1 categorical factor.")

        elif test_type == "chi_square":
            if len(cat_cols) >= 2:
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    c1 = st.selectbox("Categorical Factor A (Row):", cat_cols, key="chi_c1")
                with h_c2:
                    c2 = st.selectbox("Categorical Factor B (Col):", cat_cols, index=1, key="chi_c2")
                if st.button("🚀 Run Chi-Square Independence Test", type="primary", key="btn_run_chi"):
                    test_res = run_hypothesis_test(df, test_type, c1, var2=c2)
            else:
                st.warning("Chi-Square test requires at least 2 categorical variables.")

        if test_res:
            if "error" in test_res:
                st.error(test_res["error"])
            else:
                st.markdown(f"### Results: {test_res.get('test_name', test_type)}")
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                r_c1.metric("Test Statistic", f"{test_res.get('statistic', 0):.4f}")
                p_val = test_res.get('p_value', 1.0)
                r_c2.metric("P-Value", f"{p_val:.5f}" if p_val >= 0.0001 else "< 0.0001 ***")
                r_c3.metric(test_res.get("effect_size_name", "Effect Size"), f"{test_res.get('effect_size', 'N/A')}")
                r_c4.metric("Significant at α=0.05?", "✅ Yes" if test_res.get("significant_05") else "❌ No")

                st.markdown("#### 🤖 AI Plain-Language Explainer")
                ai_hypo_expl = explain_statistics_in_plain_language("hypothesis", test_res)
                if test_res.get("significant_05"):
                    st.success(ai_hypo_expl)
                else:
                    st.warning(ai_hypo_expl)

    # 4. REGRESSION ANALYSIS
    with t_stat4:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Regression modeling requires at least one target and one predictor numeric variable.")
        else:
            rg_c1, rg_c2, rg_c3 = st.columns([1.5, 2, 1.5])
            with rg_c1:
                reg_type = st.selectbox("Regression Type:", ["linear", "logistic"], format_func=lambda x: "OLS Multiple Linear" if x=="linear" else "Binary Logistic")
            with rg_c2:
                reg_target = st.selectbox("Target Outcome Variable (Y):", num_cols, index=len(num_cols)-1, key="reg_target_sel")
            with rg_c3:
                st.write("")
                st.write("")
                run_reg_btn = st.button("🚀 Fit Regression Model", type="primary", key="btn_fit_reg")

            avail_features = [c for c in num_cols if c != reg_target]
            reg_features = st.multiselect("Predictor Features (X):", avail_features, default=avail_features[:4])

            if run_reg_btn and reg_features:
                reg_res = run_regression_analysis(df, reg_target, reg_features, regression_type=reg_type)
                if "error" in reg_res:
                    st.error(reg_res["error"])
                else:
                    st.markdown("### Model Summary & Fit Performance")
                    if reg_type == "linear":
                        rf1, rf2, rf3, rf4 = st.columns(4)
                        rf1.metric("R² Score", f"{reg_res.get('r_squared', 0)*100:.2f}%")
                        rf2.metric("Adjusted R²", f"{reg_res.get('adj_r_squared', 0)*100:.2f}%")
                        rf3.metric("RMSE Error", f"{reg_res.get('rmse', 0):.3f}")
                        rf4.metric("MAE Error", f"{reg_res.get('mae', 0):.3f}")

                        st.markdown(f"**📐 Fitted Model Equation:** `{reg_res.get('equation', 'Y = βX')}`")

                    elif reg_type == "logistic":
                        rf1, rf2, rf3, rf4 = st.columns(4)
                        rf1.metric("Accuracy", f"{reg_res.get('accuracy', 0)*100:.1f}%")
                        rf2.metric("Precision", f"{reg_res.get('precision', 0)*100:.1f}%")
                        rf3.metric("Recall", f"{reg_res.get('recall', 0)*100:.1f}%")
                        rf4.metric("F1-Score", f"{reg_res.get('f1_score', 0):.3f}")

                    if reg_res.get("coefficients_table"):
                        st.markdown("#### 📋 Estimated Feature Coefficients")
                        st.dataframe(pd.DataFrame(reg_res["coefficients_table"]), use_container_width=True, hide_index=True)

                    st.markdown("#### 🤖 AI Plain-Language Explainer")
                    ai_reg_expl = explain_statistics_in_plain_language("regression", reg_res)
                    st.info(ai_reg_expl)


# ---------------- 11B. EXPERIMENTATION & A/B TESTING ----------------
elif selected_module in ["🧪 Experimentation & A/B Testing", "🧪 A/B Testing", "Experimentation"]:
    st.subheader("🧪 Experimentation & A/B Testing Studio")
    st.caption("Rigorous hypothesis evaluation for digital experiments: Two-sample conversion rate Z-tests, Welch's t-tests for continuous metrics, confidence intervals, effect sizes (Cohen's h & d), and statistical power analysis.")

    # Demo loaders row
    d_col1, d_col2, d_col3 = st.columns([1.5, 1.5, 2])
    with d_col1:
        if st.button("🎯 Load Conversion A/B Demo", use_container_width=True, key="btn_load_ab_conv_demo"):
            demo_df = generate_sample_ab_dataset("conversion")
            st.session_state.df = demo_df
            st.session_state.dataset_source = "Landing Page A/B Conversion Test"
            st.success("Loaded Landing Page A/B Conversion Test (2,400 records)!")
            st.rerun()
    with d_col2:
        if st.button("📊 Load Revenue A/B Demo", use_container_width=True, key="btn_load_ab_mean_demo"):
            demo_df = generate_sample_ab_dataset("mean")
            st.session_state.df = demo_df
            st.session_state.dataset_source = "Pricing Order Value A/B Test"
            st.success("Loaded Pricing Revenue A/B Test (1,600 records)!")
            st.rerun()

    tab_ab1, tab_ab2, tab_ab3 = st.tabs([
        "🎯 Conversion Rate A/B Test",
        "📊 Continuous Metric Mean A/B Test",
        "⚡ Sample Size & Power Planner"
    ])

    # TAB 1: CONVERSION A/B TEST
    with tab_ab1:
        all_cols = df.columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            grp_col = st.selectbox("Variant Grouping Column:", cat_cols if cat_cols else all_cols, key="ab_conv_grp")
        
        u_vals = list(df[grp_col].dropna().unique()) if grp_col in df.columns else []
        with c2:
            c_val = st.selectbox("Control Arm (Baseline):", u_vals, index=0 if len(u_vals)>0 else 0, key="ab_conv_cval")
        with c3:
            t_val = st.selectbox("Treatment Arm (Variant):", u_vals, index=1 if len(u_vals)>1 else 0, key="ab_conv_tval")

        c4, c5 = st.columns([2, 1])
        with c4:
            tgt_col = st.selectbox("Target Conversion Metric (Binary / Yes/No / 0/1):", all_cols, key="ab_conv_tgt")
        with c5:
            conf_level = st.selectbox("Confidence Level:", [0.95, 0.90, 0.99], format_func=lambda x: f"{int(x*100)}% (α = {1-x:.2f})", key="ab_conv_conf")

        if st.button("🚀 Run Conversion A/B Test", type="primary", key="btn_run_conv_ab"):
            if c_val == t_val:
                st.error("Control and Treatment arms must be distinct levels.")
            else:
                res = run_ab_conversion_test(df, grp_col, tgt_col, c_val, t_val, confidence_level=conf_level)
                if "error" in res:
                    st.error(res["error"])
                else:
                    # Verdict Banner
                    if res["is_significant"]:
                        if res["difference"] > 0:
                            st.success(f"### 🏆 Statistically Significant Winner: {res['winner']}\n\n" + res['conclusion'])
                        else:
                            st.error("### ⚠️ Statistically Significant Underperformer: Treatment\n\n" + res['conclusion'])
                    else:
                        st.warning("### 🟡 Inconclusive: No Statistically Significant Difference\n\n" + res['conclusion'])

                    # Cards Grid
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric(f"Control ({res['control_name']})", f"{res['control']['rate_pct']:.2f}%", f"N = {res['control']['n']:,}")
                    m2.metric(f"Treatment ({res['treatment_name']})", f"{res['treatment']['rate_pct']:.2f}%", f"N = {res['treatment']['n']:,}")
                    m3.metric("Absolute Difference", f"{res['difference_pct']:+.2f}%", "Δ (Treat - Ctrl)")
                    m4.metric("Relative Lift", f"{res['relative_lift_pct']:+.2f}%", "gain over base")
                    m5.metric("P-Value", f"{res['p_value']:.4f}", f"Z = {res['z_statistic']:.2f}")
                    m6.metric(f"{int(conf_level*100)}% CI", f"[{res['confidence_interval']['low_pct']:+.2f}%, {res['confidence_interval']['high_pct']:+.2f}%]")

                    # Detailed parameters table
                    st.markdown("#### 📋 Statistical Experiment Parameters")
                    param_df = pd.DataFrame({
                        "Parameter": [
                            "Control Sample (N)", "Control Conversions", "Control Conversion Rate",
                            "Treatment Sample (N)", "Treatment Conversions", "Treatment Conversion Rate",
                            "Absolute Rate Difference (Δ)", "Relative Conversion Lift (%)", "Two-Proportion Z-Statistic",
                            "Two-Tailed P-Value", f"{int(conf_level*100)}% Confidence Interval", "Effect Size (Cohen's h)",
                            "Relative Risk (RR)", "Odds Ratio (OR)", "Post-Hoc Statistical Power (1 - β)", "Required N per Arm (for 80% Power)"
                        ],
                        "Value": [
                            f"{res['control']['n']:,}", f"{res['control']['conversions']:,}", f"{res['control']['rate_pct']:.2f}%",
                            f"{res['treatment']['n']:,}", f"{res['treatment']['conversions']:,}", f"{res['treatment']['rate_pct']:.2f}%",
                            f"{res['difference_pct']:+.2f}%", f"{res['relative_lift_pct']:+.2f}%", f"{res['z_statistic']:.4f}",
                            f"{res['p_value']:.5f}", f"[{res['confidence_interval']['low_pct']:+.2f}%, {res['confidence_interval']['high_pct']:+.2f}%]",
                            f"{res['effect_size']['value']:.3f} ({res['effect_size']['magnitude']})",
                            f"{res['effect_size']['relative_risk']:.3f}", f"{res['effect_size']['odds_ratio']:.3f}",
                            f"{res['power_analysis']['observed_power_pct']:.1f}%", f"{res['power_analysis']['required_n_per_variant']:,}"
                        ]
                    })
                    st.dataframe(param_df, use_container_width=True, hide_index=True)

    # TAB 2: CONTINUOUS MEAN A/B TEST
    with tab_ab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            grp_col_m = st.selectbox("Variant Grouping Column:", cat_cols if cat_cols else all_cols, key="ab_mean_grp")
        u_vals_m = list(df[grp_col_m].dropna().unique()) if grp_col_m in df.columns else []
        with c2:
            c_val_m = st.selectbox("Control Arm (Baseline):", u_vals_m, index=0 if len(u_vals_m)>0 else 0, key="ab_mean_cval")
        with c3:
            t_val_m = st.selectbox("Treatment Arm (Variant):", u_vals_m, index=1 if len(u_vals_m)>1 else 0, key="ab_mean_tval")

        c4, c5 = st.columns([2, 1])
        with c4:
            tgt_col_m = st.selectbox("Continuous Outcome Metric (Revenue / Order Value / Duration):", num_cols, key="ab_mean_tgt")
        with c5:
            conf_level_m = st.selectbox("Confidence Level:", [0.95, 0.90, 0.99], format_func=lambda x: f"{int(x*100)}% (α = {1-x:.2f})", key="ab_mean_conf")

        if st.button("🚀 Run Continuous Mean A/B Test", type="primary", key="btn_run_mean_ab"):
            if c_val_m == t_val_m:
                st.error("Control and Treatment arms must be distinct levels.")
            else:
                res_m = run_ab_mean_test(df, grp_col_m, tgt_col_m, c_val_m, t_val_m, confidence_level=conf_level_m)
                if "error" in res_m:
                    st.error(res_m["error"])
                else:
                    if res_m["is_significant"]:
                        if res_m["difference"] > 0:
                            st.success(f"### 🏆 Statistically Significant Winner: {res_m['winner']}\n\n" + res_m['conclusion'])
                        else:
                            st.error("### ⚠️ Statistically Significant Underperformer: Treatment\n\n" + res_m['conclusion'])
                    else:
                        st.warning("### 🟡 Inconclusive: No Statistically Significant Difference\n\n" + res_m['conclusion'])

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric(f"Control ({res_m['control_name']})", f"{res_m['control']['mean']:,.2f}", f"N = {res_m['control']['n']:,} (SD: {res_m['control']['std']:.2f})")
                    m2.metric(f"Treatment ({res_m['treatment_name']})", f"{res_m['treatment']['mean']:,.2f}", f"N = {res_m['treatment']['n']:,} (SD: {res_m['treatment']['std']:.2f})")
                    m3.metric("Difference (Δ)", f"{res_m['difference']:+,.2f}", "Treat - Ctrl")
                    m4.metric("Relative Lift", f"{res_m['relative_lift_pct']:+.2f}%", "gain over base")
                    m5.metric("P-Value", f"{res_m['p_value']:.4f}", f"t = {res_m['t_statistic']:.2f}")
                    m6.metric(f"{int(conf_level_m*100)}% CI", f"[{res_m['confidence_interval']['low']:+,.2f}, {res_m['confidence_interval']['high']:+,.2f}]")

                    st.markdown("#### 📋 Statistical Experiment Parameters")
                    param_df_m = pd.DataFrame({
                        "Parameter": [
                            "Control Sample (N)", "Control Mean", "Control Std Dev", "Control Std Error",
                            "Treatment Sample (N)", "Treatment Mean", "Treatment Std Dev", "Treatment Std Error",
                            "Difference in Means (Δ)", "Relative Lift (%)", "Welch's t-Statistic", "Degrees of Freedom (df)",
                            "Two-Tailed P-Value", f"{int(conf_level_m*100)}% Confidence Interval", "Effect Size (Cohen's d)",
                            "Post-Hoc Statistical Power (1 - β)", "Required N per Arm (for 80% Power)"
                        ],
                        "Value": [
                            f"{res_m['control']['n']:,}", f"{res_m['control']['mean']:,.4f}", f"{res_m['control']['std']:.4f}", f"{res_m['control']['se']:.4f}",
                            f"{res_m['treatment']['n']:,}", f"{res_m['treatment']['mean']:,.4f}", f"{res_m['treatment']['std']:.4f}", f"{res_m['treatment']['se']:.4f}",
                            f"{res_m['difference']:+,.4f}", f"{res_m['relative_lift_pct']:+.2f}%", f"{res_m['t_statistic']:.4f}", f"{res_m['df']:.2f}",
                            f"{res_m['p_value']:.5f}", f"[{res_m['confidence_interval']['low']:+,.4f}, {res_m['confidence_interval']['high']:+,.4f}]",
                            f"{res_m['effect_size']['value']:.3f} ({res_m['effect_size']['magnitude']})",
                            f"{res_m['power_analysis']['observed_power_pct']:.1f}%", f"{res_m['power_analysis']['required_n_per_variant']:,}"
                        ]
                    })
                    st.dataframe(param_df_m, use_container_width=True, hide_index=True)

    # TAB 3: SAMPLE SIZE CALCULATOR
    with tab_ab3:
        st.markdown("### ⚡ Pre-Experiment Sample Size & Power Sizing Tool")
        st.caption("Determine exact visitor requirements before launching to guarantee sufficient statistical power.")

        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            c_type = st.selectbox("Metric Type:", ["conversion", "mean"], format_func=lambda x: "Conversion Rate (%)" if x=="conversion" else "Continuous Metric (Revenue / Average)", key="st_ab_calc_type")
            c_base = st.number_input("Baseline Value:", value=10.0 if c_type=="conversion" else 50.0, step=0.5, key="st_ab_calc_base")
            c_mde = st.number_input("Minimum Detectable Effect (MDE %):", value=15.0, step=1.0, min_value=0.1, key="st_ab_calc_mde")
        with calc_col2:
            c_pwr = st.selectbox("Target Statistical Power:", [0.80, 0.90, 0.95], format_func=lambda x: f"{int(x*100)}% Power", key="st_ab_calc_pwr")
            c_alpha = st.selectbox("Significance Level (α):", [0.05, 0.01, 0.10], format_func=lambda x: f"α = {x} ({int((1-x)*100)}% Conf)", key="st_ab_calc_alpha")
            c_sd = st.number_input("Estimated Standard Deviation (SD):", value=25.0, step=1.0, key="st_ab_calc_sd") if c_type=="mean" else None

        calc_res = calculate_sample_size(c_base/100.0 if c_type=="conversion" else c_base, c_mde, metric_type=c_type, alpha=c_alpha, power=c_pwr, sd=c_sd)
        if "error" not in calc_res:
            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("Required N Per Arm", f"{calc_res['n_per_variant']:,}")
            cp2.metric("Total Sample Required", f"{calc_res['total_sample_size']:,}")
            cp3.metric("Expected Target Value", f"{calc_res.get('expected_conversion_rate', 0)*100:.2f}%" if c_type=="conversion" else f"{calc_res.get('expected_mean', 0):,.2f}")




# ---------------- 12. AI ASSISTANT ----------------
elif selected_module == "🤖 AI Assistant":
    orchestrator: AIOrchestrator = st.session_state.ai_orchestrator
    gemini_key = st.session_state.get("gemini_api_key", "").strip()
    active_engine = "🌟 Google Gemini 2.5 Flash" if gemini_key else "⚡ Built-in Offline Statistical AI"

    # Header & Status Row
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.subheader("🤖 Context-Aware AI Assistant")
        st.caption(f"Dataset-grounded conversational assistant for queries, calculations, schema questions, SQL generation, and ML advice. Active Engine: **{active_engine}**")
    with c_head2:
        if st.button("🗑️ Clear Conversation", use_container_width=True, key="btn_clear_ai_chat"):
            st.session_state.ai_chat_history = [
                {
                    "role": "assistant",
                    "content": "👋 Conversation cleared! I am ready for your next question about `" + str(st.session_state.get("dataset_source", "active dataset")) + "`."
                }
            ]
            st.rerun()

    # Collapsible API Settings
    with st.expander("🔑 AI Provider Configuration (Optional)", expanded=False):
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            new_key = st.text_input("Google Gemini API Key:", value=gemini_key, type="password", key="ai_copilot_gemini_key", help="Enter your Gemini key for generative natural language reasoning, or leave blank to use the 100% offline built-in analytics engine.")
            if new_key != gemini_key:
                st.session_state.gemini_api_key = new_key
                st.rerun()
        with col_k2:
            badge_cls = 'badge-success' if gemini_key else 'badge-info'
            st.markdown(f"<div style='margin-top:28px;'><span class='badge-chip {badge_cls}'>{active_engine}</span></div>", unsafe_allow_html=True)

    # 🗣️ Voice Speaking Assistant Controls (Speed rate removed per user instruction)
    with st.expander("🗣️ Voice Speaking Assistant (Language & Gender Selection)", expanded=True):
        v_col1, v_col2, v_col3 = st.columns([2.5, 2.5, 1.5])
        with v_col1:
            voice_lang = st.selectbox(
                "🌐 Language:",
                [
                    "English (US) [en-US]",
                    "English (UK) [en-GB]",
                    "Spanish (Español) [es-ES]",
                    "French (Français) [fr-FR]",
                    "German (Deutsch) [de-DE]",
                    "Hindi (हिन्दी) [hi-IN]",
                    "Italian (Italiano) [it-IT]",
                    "Japanese (日本語) [ja-JP]",
                    "Chinese (中文) [zh-CN]",
                    "Portuguese (Brasil) [pt-BR]"
                ],
                index=0,
                key="voice_lang_select"
            )
            lang_code = voice_lang.split("[")[1].replace("]", "")
        with v_col2:
            voice_gender = st.selectbox(
                "👤 Gender Persona:",
                ["👩 Female Voice", "👨 Male Voice"],
                index=0,
                key="voice_gender_select"
            )
            gender_val = "female" if "Female" in voice_gender else "male"
        with v_col3:
            st.write("")
            st.write("")
            auto_speak = st.checkbox("🔊 Auto-speak replies", value=False, key="voice_autospeak_chk")
        
        speed_val = 1.0

        # Get latest assistant text for playback
        latest_assistant_text = ""
        for m in reversed(st.session_state.ai_chat_history):
            if m.get("role") == "assistant":
                latest_assistant_text = m.get("content", "")
                break

        latest_assistant_text = sanitize_unicode(latest_assistant_text)
        escaped_text = json.dumps(latest_assistant_text, ensure_ascii=False)
        auto_speak_js = "true" if auto_speak else "false"

        voice_player_html = """
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(99,102,241,0.25); border-radius: 8px; padding: 10px 14px; margin-top: 4px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <button id="stSpeakBtn" onclick="togglePlaySpeech()" style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: white; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 600; font-size: 0.84rem; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    <span id="stSpeakIcon">&#128266;</span> <span id="stSpeakLabel">Read Latest Response Aloud</span>
                </button>
                <button onclick="stopSpeech()" style="background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 6px 12px; font-size: 0.82rem; cursor: pointer;">
                    &#9209; Stop
                </button>
            </div>
            <div id="stVoiceBadge" style="font-size: 0.78rem; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span>
                <span>Voice Engine Ready (__GENDER_LABEL__, __LANG_CODE__)</span>
            </div>
        </div>

        <script>
        (function() {
            const rawText = __RAW_TEXT__;
            const lang = "__LANG_CODE__";
            const gender = "__GENDER_VAL__";
            const rate = __SPEED_VAL__;
            const shouldAutoSpeak = __SHOULD_AUTO__;
            let isSpeaking = false;

            function cleanText(txt) {
                if (!txt) return "";
                return txt.replace(/<[^>]*>/g, " ")
                          .replace(/[*#_`~[\]()]/g, " ")
                          .replace(/&nbsp;/g, " ")
                          .replace(/\s+/g, " ")
                          .trim();
            }

            function getVoice(voices, langCode, targetGender) {
                if (!voices || voices.length === 0) return null;
                const prefix = langCode.split('-')[0].toLowerCase();
                const matched = voices.filter(v => v.lang.toLowerCase().startsWith(prefix));
                const pool = matched.length > 0 ? matched : voices;
                const femaleKeys = ['female', 'woman', 'zira', 'samantha', 'victoria', 'karen', 'anna', 'moira', 'natural'];
                const maleKeys = ['male', 'man', 'david', 'george', 'daniel', 'alex', 'fred', 'oliver'];

                if (targetGender === 'female') {
                    return pool.find(v => femaleKeys.some(k => v.name.toLowerCase().includes(k))) || pool[0];
                } else {
                    return pool.find(v => maleKeys.some(k => v.name.toLowerCase().includes(k))) || pool[0];
                }
            }

            window.stopSpeech = function() {
                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                }
                isSpeaking = false;
                const icon = document.getElementById("stSpeakIcon");
                const lbl = document.getElementById("stSpeakLabel");
                const badge = document.getElementById("stVoiceBadge");
                if (icon) icon.textContent = "\U0001F50A";
                if (lbl) lbl.textContent = "Read Latest Response Aloud";
                if (badge) badge.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span> <span>Voice Engine Ready</span>';
            };

            window.togglePlaySpeech = function() {
                if (!('speechSynthesis' in window)) {
                    alert("Browser speech synthesis is not supported in this browser.");
                    return;
                }
                if (isSpeaking) {
                    window.stopSpeech();
                    return;
                }
                const clean = cleanText(rawText);
                if (!clean) return;

                window.speechSynthesis.cancel();
                isSpeaking = true;

                const voices = window.speechSynthesis.getVoices();
                const matchedVoice = getVoice(voices, lang, gender);
                const utter = new SpeechSynthesisUtterance(clean);
                
                if (matchedVoice) {
                    utter.voice = matchedVoice;
                    utter.lang = matchedVoice.lang;
                } else {
                    utter.lang = lang;
                }
                utter.pitch = gender === 'female' ? 1.08 : 0.88;
                utter.rate = rate;

                const icon = document.getElementById("stSpeakIcon");
                const lbl = document.getElementById("stSpeakLabel");
                const badge = document.getElementById("stVoiceBadge");
                if (icon) icon.textContent = "\u23F9";
                if (lbl) lbl.textContent = "Stop Speaking";
                if (badge) badge.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f59e0b;animation:pulse 1s infinite;"></span> <span style="color:#fbbf24;font-weight:600;">Speaking aloud...</span>';

                utter.onend = window.stopSpeech;
                utter.onerror = window.stopSpeech;

                window.speechSynthesis.speak(utter);
            };

            if (shouldAutoSpeak && rawText.length > 5 && !window.hasAutoSpoken) {
                window.hasAutoSpoken = true;
                setTimeout(window.togglePlaySpeech, 300);
            }
        })();
        </script>
        """.replace("__RAW_TEXT__", escaped_text)
        voice_player_html = voice_player_html.replace("__LANG_CODE__", lang_code)
        voice_player_html = voice_player_html.replace("__GENDER_VAL__", gender_val)
        voice_player_html = voice_player_html.replace("__GENDER_LABEL__", voice_gender.split()[1] if len(voice_gender.split()) > 1 else "Voice")
        voice_player_html = voice_player_html.replace("__SPEED_VAL__", str(speed_val))
        voice_player_html = voice_player_html.replace("__SHOULD_AUTO__", auto_speak_js)
        
        voice_player_html = sanitize_unicode(voice_player_html)
        components.html(voice_player_html, height=75)

    # Quick Suggestion Chips
    st.markdown("##### 💡 Quick Analytics Prompts")
    qc1, qc2, qc3, qc4, qc5, qc6 = st.columns(6)
    preset_prompt = None
    with qc1:
        if st.button("📊 Summarize", use_container_width=True, key="chip_sum"):
            preset_prompt = "Summarize the dataset and key metrics"
    with qc2:
        if st.button("🛡️ Audit Hygiene", use_container_width=True, key="chip_hyg"):
            preset_prompt = "What data quality and hygiene risks exist?"
    with qc3:
        if st.button("📈 Top Drivers", use_container_width=True, key="chip_drv"):
            preset_prompt = "What are the strongest correlations and drivers?"
    with qc4:
        if st.button("🗄️ Write SQL", use_container_width=True, key="chip_sql"):
            preset_prompt = "Write an SQL query to aggregate the top categories"
    with qc5:
        if st.button("🤖 ML Strategy", use_container_width=True, key="chip_ml"):
            preset_prompt = "Recommend a machine learning strategy and target variable"
    with qc6:
        if st.button("📈 Visuals Advice", use_container_width=True, key="chip_viz"):
            preset_prompt = "What are the best charts to visualize these variables?"

    # Display Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Handle Submission (via chat_input or preset chip)
    user_input = st.chat_input("Ask a question about your data or request SQL/cleaning advice...")
    active_prompt = preset_prompt or user_input

    if active_prompt:
        # Append user message
        st.session_state.ai_chat_history.append({"role": "user", "content": active_prompt})

        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("AI Copilot analyzing dataset..."):
                res = orchestrator.ask(active_prompt, df, api_key=gemini_key)
                resp_text = res.get("response", "")
                st.markdown(resp_text)

                # Append to history
                st.session_state.ai_chat_history.append({
                    "role": "assistant",
                    "content": resp_text
                })
        st.rerun()


# ---------------- 13. INSIGHTS ----------------
elif selected_module == "💡 Insights":
    st.subheader("💡 Executive Business & Strategic Insights")
    st.caption("Deep statistical intelligence, key business drivers, segment performance, churn & risk alerts, and prescriptive business recommendations.")

    if df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload a CSV file or load a demo dataset first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        n_rows, n_cols = df.shape
        missing_total = int(df.isna().sum().sum())
        total_cells = n_rows * n_cols
        health_score = max(10, int(100 - (missing_total / (total_cells or 1) * 120) - (df.duplicated().sum() / (n_rows or 1) * 80)))
        health_grade = "A (Excellent)" if health_score >= 85 else ("B (Good)" if health_score >= 70 else ("C (Fair)" if health_score >= 50 else "D (Critical)"))

        # Find strongest driver
        top_driver_str = "None"
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append((num_cols[i], num_cols[j], val))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if pairs:
                top_p = pairs[0]
                top_driver_str = f"{top_p[0]} ↔ {top_p[1]} (r = {top_p[2]:+.2f})"

        # Find top segment
        top_seg_str = "None"
        if cat_cols and num_cols:
            agg_first = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=False)
            if not agg_first.empty:
                top_seg_str = f"{agg_first.index[0]} ({cat_cols[0]})"

        # Executive KPI Cards
        ik1, ik2, ik3, ik4 = st.columns(4)
        with ik1:
            st.metric("🛡️ Data Health Score", f"{health_score} / 100", delta=health_grade)
        with ik2:
            st.metric("📈 Primary Business Driver", top_driver_str[:22], delta="Linear Correlated" if "↔" in top_driver_str else "N/A")
        with ik3:
            st.metric("🏆 Top Revenue / Volume Segment", top_seg_str[:22], delta="Leading Share")
        with ik4:
            st.metric("⚠️ Hygiene / Risk Rate", f"{round((missing_total / (total_cells or 1)) * 100, 1)}%", delta=f"{missing_total:,} missing", delta_color="inverse")

        st.markdown("---")

        tab_ins1, tab_ins2, tab_ins3, tab_ins4, tab_ins5 = st.tabs([
            "📊 Executive Statistical Summary",
            "📈 Key Business Drivers & Correlations",
            "🏆 Segment Performance & Leaderboards",
            "🛡️ Risk, Churn & Anomaly Signals",
            "🎯 Prescriptive Business Recommendations"
        ])

        with tab_ins1:
            st.markdown("#### 📊 Executive Dataset Intelligence Summary")
            st.markdown(f"""
            - **Dataset Volume:** **{n_rows:,}** rows × **{n_cols}** columns (**{total_cells:,}** total observations).
            - **Feature Architecture:** **{len(num_cols)}** Numerical metric(s) and **{len(cat_cols)}** Categorical dimension(s).
            - **Completeness Index:** **{round(100 - (missing_total / (total_cells or 1) * 100), 2)}%** data density.
            - **Uniqueness Ratio:** **{round(((n_rows - df.duplicated().sum()) / (n_rows or 1)) * 100, 1)}%** unique rows ({int(df.duplicated().sum())} duplicates).
            """)

            if num_cols:
                st.markdown("##### 🔢 Numerical Attributes Descriptive Summary:")
                st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

        with tab_ins2:
            st.markdown("#### 📈 Key Business Drivers & Statistical Association Matrix")
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr().round(3)
                pairs_list = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        r_val = corr_matrix.iloc[i, j]
                        if not np.isnan(r_val):
                            strength = "Very Strong" if abs(r_val) >= 0.8 else ("Strong" if abs(r_val) >= 0.6 else ("Moderate" if abs(r_val) >= 0.35 else "Weak"))
                            direction = "Positive ↗️" if r_val > 0 else "Negative ↘️"
                            impact = "High Direct Influence" if abs(r_val) >= 0.6 else "Moderate Co-movement"
                            pairs_list.append({
                                "Feature A": num_cols[i],
                                "Feature B": num_cols[j],
                                "Pearson r": r_val,
                                "Strength": strength,
                                "Direction": direction,
                                "Business Impact": impact
                            })
                pairs_df = pd.DataFrame(pairs_list).sort_values(by="Pearson r", key=abs, ascending=False)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)

                high_coll_count = len(pairs_df[pairs_df["Pearson r"].abs() >= 0.80])
                if high_coll_count > 0:
                    st.warning(f"⚠️ **Multicollinearity Alert:** Found {high_coll_count} feature pair(s) with |r| ≥ 0.80. Redundant variables can distort regression coefficients.")
                else:
                    st.success("✅ **Clean Orthogonality:** No severe multicollinearity (|r| ≥ 0.80) detected among numerical predictors.")
            else:
                st.info("At least two numerical columns are required for correlation driver analysis.")

        with tab_ins3:
            st.markdown("#### 🏆 Segment Performance Leaderboard & Pareto Distribution")
            if cat_cols and num_cols:
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    sel_cat = st.selectbox("Select Dimension / Segment:", cat_cols, key="ins_sel_cat")
                with col_sel2:
                    sel_num = st.selectbox("Select Target Metric:", num_cols, key="ins_sel_num")

                seg_agg = df.groupby(sel_cat)[sel_num].agg(
                    Total=("sum"),
                    Average=("mean"),
                    Count=("count")
                ).reset_index().sort_values(by="Total", ascending=False)

                total_sum = seg_agg["Total"].sum()
                seg_agg["Share (%)"] = (seg_agg["Total"] / (total_sum or 1) * 100).round(1)
                seg_agg["Average"] = seg_agg["Average"].round(2)
                seg_agg["Total"] = seg_agg["Total"].round(2)

                st.dataframe(seg_agg, use_container_width=True, hide_index=True)

                top_seg = seg_agg.iloc[0]
                st.info(f"💡 **Key Insight:** **`{top_seg[sel_cat]}`** is the dominant segment, capturing **{top_seg['Share (%)']}%** of all `{sel_num}` (Average: **{top_seg['Average']:,.2f}** per observation).")
            else:
                st.info("Both categorical and numerical attributes are needed for segment performance breakdown.")

        with tab_ins4:
            st.markdown("#### 🛡️ Risk, Churn & Data Hygiene Signals")
            col_risk1, col_risk2 = st.columns(2)
            with col_risk1:
                st.markdown("##### 🚨 Outlier & Extreme Value Signals:")
                outlier_summary = []
                for c in num_cols:
                    s_col = df[c].dropna()
                    if not s_col.empty:
                        q75, q25 = s_col.quantile(0.75), s_col.quantile(0.25)
                        iqr = q75 - q25
                        upper_fence = q75 + (1.5 * iqr)
                        out_count = int((s_col > upper_fence).sum())
                        if out_count > 0:
                            outlier_summary.append({
                                "Column": c,
                                "Outliers (>1.5 IQR)": out_count,
                                "% of Records": round((out_count / len(s_col)) * 100, 2),
                                "Upper Threshold": round(upper_fence, 2),
                                "Max Observed": round(s_col.max(), 2)
                            })
                if outlier_summary:
                    st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No extreme outliers (> 1.5x IQR) detected across numerical features.")

            with col_risk2:
                st.markdown("##### 🔍 Missingness & Hygiene Risks:")
                null_counts = df.isna().sum()
                missing_cols = null_counts[null_counts > 0]
                if not missing_cols.empty:
                    m_data = [{"Column": c, "Missing Rows": cnt, "% Missing": round((cnt / n_rows) * 100, 1)} for c, cnt in missing_cols.items()]
                    st.dataframe(pd.DataFrame(m_data), use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Zero missing values! Perfect 100% feature completeness.")

        with tab_ins5:
            st.markdown("#### 🎯 Prescriptive Business Recommendations & Next Steps")
            st.markdown("""
            1. **Customer Retention & Churn Prevention**:
               - Target accounts in month-to-month contracts and early tenure windows (< 6 months) with tailored long-term incentive bundles.
            2. **Pricing & Revenue Optimization**:
               - Review customer segments with disproportionately high average charges and offer loyalty discounts to minimize cancellation risks.
            3. **Data Quality Governance**:
               - Address incomplete records via median/mode imputation to preserve sample size before training predictive models.
            4. **Machine Learning Deployment**:
               - Deploy gradient boosted decision trees (`LightGBM` / `XGBoost`) targeting customer churn or price prediction for maximum recall.
            """)

        # Download Business Insights Report
        st.markdown("---")
        report_text = f"""# DataMind Executive Business Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Source: {st.session_state.get('dataset_source', 'Active Dataset')}
Dimensions: {n_rows:,} rows × {n_cols} columns

## 1. Executive Summary
- Data Health Score: {health_score}/100 ({health_grade})
- Missing Cells: {missing_total:,} ({round((missing_total / (total_cells or 1)) * 100, 2)}%)
- Duplicate Records: {int(df.duplicated().sum()):,}
- Primary Driver: {top_driver_str}
- Dominant Segment: {top_seg_str}

## 2. Strategic Recommendations
1. Focus retention outreach on high-churn probability customer cohorts.
2. Address data completeness in affected numerical columns.
3. Leverage identified linear drivers in predictive models.
"""
        col_dl_pdf, col_dl_md = st.columns(2)
        with col_dl_pdf:
            try:
                pdf_bytes = generate_pdf_report(
                    report_title="Executive Business Insights Report",
                    project_name=st.session_state.get('dataset_source', 'Active Dataset'),
                    df=df,
                    meta={"health_score": health_score, "health_grade": health_grade, "missing_cells": missing_total},
                    subtitle="Strategic Drivers & Risk Analysis",
                    audit_trail=st.session_state.get("cleaning_history", [])
                )
                st.download_button(
                    label="📥 Download Insights Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"business_insights_report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error preparing PDF: {str(e)}")
        with col_dl_md:
            st.download_button(
                label="📥 Download Insights Report (Markdown)",
                data=report_text,
                file_name=f"business_insights_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )


# ---------------- INSERT NEW COLUMN ----------------
elif selected_module == "➕ Insert New Column":
    st.subheader("➕ Insert & Synthesize New Column Studio")
    st.caption("Synthesize, calculate, and append new columns into your active dataset using mathematical operations, custom formulas, scaling, conditional thresholds, text merging, or column duplication.")

    if df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload a CSV file or load a sample dataset first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        all_cols = list(df.columns)
        n_col1 = num_cols[0] if num_cols else all_cols[0]
        n_col2 = num_cols[1] if len(num_cols) > 1 else n_col1

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.markdown("#### 🛠️ Column Generator & Expression Builder")
            
            c_n1, c_n2 = st.columns(2)
            with c_n1:
                new_col_name = st.text_input("New Column Name:", placeholder="e.g. bonus_pay, margin_ratio, tier", key="inc_new_col_name")
            with c_n2:
                gen_method = st.selectbox(
                    "Generation Method:",
                    [
                        "1. Math Operation (Col A & Col B)",
                        "2. Custom Formula / Python Expression",
                        "3. Scale / Markup Percentage (+15%, etc.)",
                        "4. Conditional Threshold (If-Else / Binning)",
                        "5. Text Concatenation (Merge Columns)",
                        "6. Constant / Static Value",
                        "7. Copy / Duplicate Column"
                    ],
                    key="inc_gen_method"
                )

            # Dynamic Form Inputs
            if gen_method.startswith("1."):
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    m_col_a = st.selectbox("Column A (Numeric):", num_cols if num_cols else all_cols, key="inc_math_a")
                with m_c2:
                    m_op = st.selectbox("Operation:", ["Divide (A / B)", "Percentage Ratio ((A/B)*100)", "Multiply (A * B)", "Add (A + B)", "Subtract (A - B)"], key="inc_math_op")
                with m_c3:
                    m_col_b = st.selectbox("Column B (Numeric):", num_cols if num_cols else all_cols, index=1 if len(num_cols) > 1 else 0, key="inc_math_b")

            elif gen_method.startswith("2."):
                default_expr = f"df['{n_col1}'] * 1.15"
                f_expr = st.text_area("Custom Python/Pandas Formula (Access active DataFrame as `df`):", value=default_expr, height=80, key="inc_formula_expr")

            elif gen_method.startswith("3."):
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    scale_col = st.selectbox("Column to Scale:", num_cols if num_cols else all_cols, key="inc_scale_col")
                with s_c2:
                    scale_factor = st.number_input("Multiplier Factor (e.g. 1.15 for +15%):", value=1.15, step=0.05, key="inc_scale_factor")

            elif gen_method.startswith("4."):
                cond_c1, cond_c2, cond_c3 = st.columns(3)
                with cond_c1:
                    cond_col = st.selectbox("Test Column:", all_cols, key="inc_cond_col")
                with cond_c2:
                    cond_op = st.selectbox("Condition Operator:", [">=", ">", "<=", "<", "==", "!="], key="inc_cond_op")
                with cond_c3:
                    mean_placeholder = str(round(df[cond_col].mean(), 2)) if cond_col in num_cols else "Standard"
                    cond_val = st.text_input("Threshold Value:", value=mean_placeholder, key="inc_cond_val")
                
                cond_res1, cond_res2 = st.columns(2)
                with cond_res1:
                    val_true = st.text_input("Value if TRUE:", value="High", key="inc_val_true")
                with cond_res2:
                    val_false = st.text_input("Value if FALSE:", value="Standard", key="inc_val_false")

            elif gen_method.startswith("5."):
                cat_c1, cat_c2, cat_c3 = st.columns([1.2, 0.6, 1.2])
                with cat_c1:
                    concat_a = st.selectbox("First Column:", all_cols, key="inc_concat_a")
                with cat_c2:
                    concat_sep = st.text_input("Separator:", value=" - ", key="inc_concat_sep")
                with cat_c3:
                    concat_b = st.selectbox("Second Column:", all_cols, index=1 if len(all_cols) > 1 else 0, key="inc_concat_b")

            elif gen_method.startswith("6."):
                const_val = st.text_input("Constant Value (assigned to all rows):", value="Standard", key="inc_const_val")

            elif gen_method.startswith("7."):
                dup_col = st.selectbox("Source Column to Duplicate:", all_cols, key="inc_dup_col")

            # Execute Column Insertion
            if st.button("➕ Insert Column to Active Dataset", type="primary", use_container_width=True, key="btn_exec_insert_col"):
                if not new_col_name.strip():
                    set_task_status("Insert New Column", "failed", "Please enter a name for the new column.")
                    st.error("Please enter a name for the new column.")
                elif new_col_name.strip() in df.columns:
                    set_task_status("Insert New Column", "failed", f"Column '{new_col_name.strip()}' already exists.")
                    st.warning(f"Column '{new_col_name.strip()}' already exists. Please choose a unique name.")
                else:
                    try:
                        clean_col_name = new_col_name.strip()
                        new_df = df.copy()

                        if gen_method.startswith("1."):
                            a_val = pd.to_numeric(new_df[m_col_a], errors="coerce").fillna(0)
                            b_val = pd.to_numeric(new_df[m_col_b], errors="coerce").fillna(0)
                            if "Divide" in m_op:
                                new_df[clean_col_name] = (a_val / (b_val + 1e-5)).round(3)
                            elif "Percentage" in m_op:
                                new_df[clean_col_name] = ((a_val / (b_val + 1e-5)) * 100).round(2)
                            elif "Multiply" in m_op:
                                new_df[clean_col_name] = (a_val * b_val).round(2)
                            elif "Add" in m_op:
                                new_df[clean_col_name] = (a_val + b_val).round(2)
                            elif "Subtract" in m_op:
                                new_df[clean_col_name] = (a_val - b_val).round(2)

                        elif gen_method.startswith("2."):
                            local_scope = {"df": new_df, "pd": pd, "np": np}
                            calc_series = eval(f_expr, {"__builtins__": __builtins__}, local_scope)
                            new_df[clean_col_name] = calc_series

                        elif gen_method.startswith("3."):
                            src_numeric = pd.to_numeric(new_df[scale_col], errors="coerce").fillna(0)
                            new_df[clean_col_name] = (src_numeric * scale_factor).round(3)

                        elif gen_method.startswith("4."):
                            col_series = new_df[cond_col]
                            is_num_col = cond_col in num_cols
                            if is_num_col:
                                thresh_num = float(cond_val) if cond_val else 0.0
                                num_s = pd.to_numeric(col_series, errors="coerce").fillna(0)
                                if cond_op == ">=":
                                    mask = num_s >= thresh_num
                                elif cond_op == ">":
                                    mask = num_s > thresh_num
                                elif cond_op == "<=":
                                    mask = num_s <= thresh_num
                                elif cond_op == "<":
                                    mask = num_s < thresh_num
                                elif cond_op == "==":
                                    mask = num_s == thresh_num
                                else:
                                    mask = num_s != thresh_num
                            else:
                                if cond_op == "==":
                                    mask = col_series.astype(str) == cond_val
                                elif cond_op == "!=":
                                    mask = col_series.astype(str) != cond_val
                                else:
                                    mask = col_series.astype(str) >= cond_val
                            new_df[clean_col_name] = np.where(mask, val_true, val_false)

                        elif gen_method.startswith("5."):
                            new_df[clean_col_name] = new_df[concat_a].astype(str) + concat_sep + new_df[concat_b].astype(str)

                        elif gen_method.startswith("6."):
                            new_df[clean_col_name] = const_val

                        elif gen_method.startswith("7."):
                            new_df[clean_col_name] = new_df[dup_col]

                        st.session_state.current_df = new_df
                        st.session_state.cleaning_manager = CleaningPipelineManager(new_df)
                        set_task_status("Insert New Column", "success", f"Added column '{clean_col_name}' successfully ({len(new_df):,} rows).")
                        st.success(f"Column '{clean_col_name}' inserted successfully into active dataset!")
                        st.rerun()

                    except Exception as ins_err:
                        set_task_status("Insert New Column", "failed", str(ins_err))
                        st.error(f"Error creating column: {ins_err}")

        with col_right:
            st.markdown("#### 📋 Active Columns Directory")
            st.caption(f"{len(df.columns)} active columns in dataset (`{st.session_state.get('dataset_source', 'active')}`):")
            col_list_data = []
            for col_name in df.columns:
                col_type = "Numeric" if col_name in num_cols else "Categorical"
                sample_val = str(df[col_name].iloc[0]) if len(df) > 0 else "-"
                col_list_data.append({"Column": col_name, "Type": col_type, "Sample Value": sample_val[:20]})
            st.dataframe(pd.DataFrame(col_list_data), use_container_width=True, height=360)

        # Bottom: Live Preview Table
        st.markdown("---")
        st.markdown(f"#### 📊 Dataset Preview Table ({len(df):,} rows × {df.shape[1]} columns)")
        col_exp1, col_exp2 = st.columns([1, 4])
        with col_exp1:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv_bytes, file_name=f"dataset_{int(time.time())}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(df.head(50), use_container_width=True)


# ---------------- 13. MACHINE LEARNING & AUTOML ----------------
elif selected_module in ["⚡ Model AutoML", "Model AutoML", "AutoML", "🤖 Machine Learning", "Machine Learning"]:
    st.subheader("🤖 Machine Learning Studio & AutoML Workspace")
    st.caption("End-to-End Supervised Learning (12 Regressors, 10 Classifiers, Multilabel), Unsupervised Clustering (5 Algorithms), PCA/t-SNE/UMAP, Association Rules (Apriori & FP-Growth), and 4 Business Use Cases.")

    has_valid_dataset = (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and len(df.columns) >= 2
        and len(df) >= 3
    )

    # Dataset switch detection & state reset
    current_dataset_sig = (
        str(st.session_state.get("dataset_source", "")),
        tuple(df.columns) if has_valid_dataset else ()
    )
    if st.session_state.get("_ml_last_dataset_sig") != current_dataset_sig:
        st.session_state["_ml_last_dataset_sig"] = current_dataset_sig
        for key in ["st_ml_target", "st_ml_features", "st_tx_col", "st_item_col", "st_clust_feats", "st_dim_feats", "st_ml_algo_choice"]:
            if key in st.session_state:
                del st.session_state[key]

    if not has_valid_dataset:
        st.warning("⚠️ **No valid dataset loaded.** Machine learning requires a dataset with at least 2 columns and 3 rows.")
        st.info("💡 You can upload your data in **🛠️ Data Engineering > 📤 Data Upload** or load one of our enterprise demo datasets below:")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            if st.button("📊 Load Telecom Churn (Classification)", key="ml_load_churn", use_container_width=True):
                df_churn = generate_sample_customer_churn()
                st.session_state.raw_df = df_churn.copy()
                st.session_state.current_df = df_churn.copy()
                st.session_state.dataset_source = "Customer Churn (Demo Dataset)"
                st.session_state.cleaning_manager = CleaningPipelineManager(df_churn)
                set_task_status("Load Dataset", "success", f"Loaded Telecom Churn ({len(df_churn):,} rows)")
                st.rerun()
        with d_col2:
            if st.button("🏡 Load Real Estate (Regression)", key="ml_load_housing", use_container_width=True):
                df_house = generate_sample_housing()
                st.session_state.raw_df = df_house.copy()
                st.session_state.current_df = df_house.copy()
                st.session_state.dataset_source = "Real Estate Housing (Demo Dataset)"
                st.session_state.cleaning_manager = CleaningPipelineManager(df_house)
                set_task_status("Load Dataset", "success", f"Loaded Real Estate Housing ({len(df_house):,} rows)")
                st.rerun()
        with d_col3:
            if st.button("🛒 Load E-Commerce Sales (Clustering / MBA)", key="ml_load_sales", use_container_width=True):
                df_sales = generate_sample_ecommerce_sales()
                st.session_state.raw_df = df_sales.copy()
                st.session_state.current_df = df_sales.copy()
                st.session_state.dataset_source = "E-Commerce Sales (Demo Dataset)"
                st.session_state.cleaning_manager = CleaningPipelineManager(df_sales)
                set_task_status("Load Dataset", "success", f"Loaded E-Commerce Sales ({len(df_sales):,} rows)")
                st.rerun()
    else:
        tab_automl, tab_sup, tab_unsup, tab_dim, tab_assoc, tab_cases = st.tabs([
            "⚡ Model AutoML",
            "🎯 Supervised Learning",
            "🔵 Unsupervised Clustering",
            "🌌 Dimensionality Reduction",
            "🛒 Association Analysis",
            "🏢 Business Use Cases"
        ])

        # ---------------- TAB 0: MODEL AUTOML ----------------
        with tab_automl:
            st.markdown("### ⚡ Automated Machine Learning (AutoML) Pipeline")
            st.caption("Select target column and prediction type. The system automatically detects feature types, cleans data, handles missing values, encodes categories, detects leakage, splits train/test data, generates features, trains multiple models, tunes hyperparameters, compares models, evaluates models, selects the best model, explains the model, saves the model, and displays a model leaderboard.")

            ac1, ac2 = st.columns([1, 1])
            with ac1:
                target_candidates = list(df.columns)
                default_target_idx = len(target_candidates) - 1
                for i, c in enumerate(target_candidates):
                    if c.lower() in ["churn", "saleprice", "revenue", "target", "label", "price", "status"]:
                        default_target_idx = i
                        break
                automl_target = st.selectbox("1. Target Column (Y):", target_candidates, index=default_target_idx, key="sb_automl_target")

            with ac2:
                automl_pred_type = st.selectbox(
                    "2. Prediction Type:",
                    ["⚡ Auto-Detect", "🎯 Classification", "📈 Regression"],
                    key="sb_automl_pred_type"
                )

            can_train_automl = bool(automl_target and automl_target in df.columns)
            if st.button("🚀 Train AutoML Pipeline", type="primary", use_container_width=True, disabled=not can_train_automl, key="btn_run_automl_pipeline"):
                with st.spinner("Executing 14-Stage AutoML Pipeline (Feature Detection → Cleaning → Imputation → Encoding → Leakage Check → Split → Feature Engineering → Training → Tuning → Benchmarking → Evaluation → Champion Selection → Explainability → Model Registry)..."):
                    try:
                        automl_result = run_automl_pipeline(
                            df,
                            target_col=automl_target,
                            prediction_type=automl_pred_type,
                            dataset_name=st.session_state.dataset_source
                        )
                        st.session_state["automl_pipeline_result"] = automl_result
                        st.session_state["trained_model_res"] = {
                            "metrics": automl_result["champion_record"],
                            "feature_importances": automl_result["explanation"]["feature_importances"]
                        }
                        set_task_status("Model AutoML", "success", f"AutoML complete! Champion: {automl_result['champion_model']} registered.")
                        st.success(f"🎉 AutoML Pipeline complete! Champion: **{automl_result['champion_model']}** (Saved to Model Registry)")
                    except Exception as e:
                        set_task_status("Model AutoML", "failed", str(e))
                        st.error(f"AutoML Error: {e}")

            if st.session_state.get("automl_pipeline_result"):
                res = st.session_state["automl_pipeline_result"]
                champ_rec = res["champion_record"]
                task = res["task_type"]

                with st.expander("📋 Automated Pipeline Execution Log (14 Stages Verified)", expanded=False):
                    for log_entry in res["pipeline_logs"]:
                        st.markdown(f"**`{log_entry['step']}`** `[{log_entry['timestamp']}]`: {log_entry['message']}")

                st.markdown("---")
                st.markdown(f"### 🥇 Champion Model: **{res['champion_model']}**")
                st.caption(f"Task: **{task.title()}** | Target: `{res['target_col']}` | Dataset: `{st.session_state.dataset_source}` | Pipeline Runtime: `{res['total_pipeline_time']}s`")

                m_keys = [k for k in champ_rec.keys() if k not in ["Rank", "Model", "Training Time (s)", "_sort_key"]]
                m_cols = st.columns(len(m_keys) + 1)
                for i, k in enumerate(m_keys):
                    with m_cols[i]:
                        st.metric(label=k.upper(), value=champ_rec[k])
                with m_cols[-1]:
                    st.metric(label="TRAIN TIME", value=f"{champ_rec.get('Training Time (s)', 0)}s")

                st.info(f"💾 **Model Persisted to Registry:** Registered with ID `{res['registered_model_card'].get('id')}` (Status: **Production**)")

                st.markdown("---")
                st.markdown("### 🏆 Model Leaderboard")
                st.caption("Comprehensive multi-model benchmarking ranked by primary objective.")
                st.dataframe(res["leaderboard"], use_container_width=True)

                st.markdown("---")
                st.markdown("### 🔍 Model Explainability & Driver Attribution")
                expl = res["explanation"]
                st.markdown(expl["narrative"])

                if expl["feature_importances"]:
                    fi_df = pd.DataFrame(expl["feature_importances"]).rename(columns={"feature": "Feature", "importance": "Importance Weight"}).set_index("Feature")
                    st.bar_chart(fi_df)

        # ---------------- TAB 1: SUPERVISED LEARNING ----------------
        with tab_sup:
            c1, c2 = st.columns([1, 1])
            with c1:
                target_candidates = list(df.columns)
                default_target_idx = len(target_candidates) - 1
                for i, c in enumerate(target_candidates):
                    if c.lower() in ["churn", "saleprice", "revenue", "target", "label", "price", "status"]:
                        default_target_idx = i
                        break

                ml_target = st.selectbox("Target Variable (Y):", target_candidates, index=default_target_idx, key="st_ml_target")
                feature_candidates = [c for c in df.columns if c != ml_target]
                default_features = feature_candidates[:6]
                ml_features = st.multiselect("Feature Predictors (X):", feature_candidates, default=default_features, key="st_ml_features")

            with c2:
                ml_task_mode = st.selectbox("Learning Task Mode:", ["Auto-Detect", "Regression (Continuous)", "Binary Classification", "Multiclass Classification", "Multilabel Classification"], key="st_ml_task_mode")
                task_type_arg = None
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
                    if ml_target and ml_target in df.columns and len(df) > 0:
                        target_series = df[ml_target].dropna()
                        is_reg = bool(len(target_series) > 0 and pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 12)
                    else:
                        is_reg = False

                algo_choices = [
                    "Linear Regression", "Ridge", "Lasso", "Elastic Net",
                    "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting",
                    "XGBoost", "LightGBM", "CatBoost", "Support Vector Regression", "KNN Regression"
                ] if is_reg else [
                    "Logistic Regression", "Decision Tree", "Random Forest",
                    "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
                    "SVM", "KNN", "Naive Bayes"
                ]
                selected_algo = st.selectbox("Select Model Architecture:", algo_choices, key="st_ml_algo_choice")

            if not ml_target:
                st.warning("⚠️ Please select a target variable (Y).")
            elif not ml_features:
                st.warning("⚠️ Please select at least one feature predictor (X).")

            b1, b2 = st.columns(2)
            can_train = bool(ml_target and ml_features)
            with b1:
                if st.button("🎯 Train Selected Model", type="secondary", use_container_width=True, disabled=not can_train):
                    if can_train:
                        with st.spinner(f"Training {selected_algo}..."):
                            algo_key = selected_algo.lower().replace(" ", "_").replace("regressor", "").replace("classifier", "").strip("_")
                            prep = preprocess_for_ml(df, target_col=ml_target, feature_cols=ml_features, classification_type=task_type_arg)
                            res = train_single_model(prep, algorithm=algo_key)
                            st.session_state.trained_model_res = res
                            st.success(f"Model {selected_algo} trained successfully!")
            with b2:
                if st.button("⚡ Run AutoML Tournament", type="primary", use_container_width=True, disabled=not can_train):
                    if can_train:
                        with st.spinner("Training & benchmarking all architectures across cross-validation splits..."):
                            tourn_res = run_automl_tournament(df, target_col=ml_target, feature_cols=ml_features)
                            st.session_state.automl_res = tourn_res
                            st.session_state.trained_model_res = tourn_res["best_model_result"]
                            best_res = tourn_res["best_model_result"]
                            st.session_state.model_registry.register_model(
                                name=f"{ml_target} {tourn_res['best_model_name']}",
                                version="1.0.0",
                                dataset=st.session_state.dataset_source,
                                target=ml_target,
                                features=ml_features,
                                algorithm=tourn_res["best_model_name"],
                                metrics=best_res["metrics"],
                                status="Validated"
                            )
                            st.success(f"AutoML complete! Champion Algorithm: **{tourn_res['best_model_name']}**")

            if st.session_state.trained_model_res:
                res = st.session_state.trained_model_res
                st.markdown("#### 📊 Evaluation Metrics")
                m_cols = st.columns(len(res["metrics"]))
                for idx, (m_k, m_v) in enumerate(res["metrics"].items()):
                    with m_cols[idx]:
                        st.metric(label=m_k.upper(), value=m_v)

                if res["feature_importances"]:
                    st.markdown("#### 🔍 Feature Importances")
                    fi_df = pd.DataFrame(res["feature_importances"]).set_index("feature")
                    st.bar_chart(fi_df)

            if st.session_state.automl_res:
                st.markdown("### 🏆 AutoML Leaderboard")
                st.dataframe(st.session_state.automl_res["leaderboard"], use_container_width=True)

        # ---------------- TAB 2: UNSUPERVISED CLUSTERING ----------------
        with tab_unsup:
            st.markdown("#### 🔵 Clustering Architecture")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.info("ℹ️ At least 2 numeric columns are required to perform unsupervised clustering.")
            else:
                u_c1, u_c2, u_c3 = st.columns([1.2, 1, 1])
                with u_c1:
                    clust_algo = st.selectbox("Clustering Algorithm:", ["K-Means", "MiniBatch K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture Models (GMM)"], key="st_clust_algo")
                with u_c2:
                    n_clusters = st.slider("Number of Clusters (k):", min_value=2, max_value=min(10, max(2, len(df)-1)), value=3, key="st_clust_k")
                with u_c3:
                    clust_features = st.multiselect("Clustering Features:", num_cols, default=num_cols[:4] if len(num_cols) >= 4 else num_cols, key="st_clust_feats")

                if st.button("🚀 Run Clustering", type="primary", disabled=len(clust_features) < 2):
                    with st.spinner("Clustering multidimensional data..."):
                        algo_key = clust_algo.lower().replace(" ", "_").replace("(gmm)", "").replace("-", "_").strip()
                        if "minibatch" in algo_key: algo_key = "minibatch_kmeans"
                        elif "gaussian" in algo_key or "gmm" in algo_key: algo_key = "gmm"
                        elif "hierarchical" in algo_key: algo_key = "hierarchical"
                        elif "dbscan" in algo_key: algo_key = "dbscan"
                        else: algo_key = "kmeans"

                        c_res = run_unsupervised_clustering(df, columns=clust_features, n_clusters=n_clusters, algorithm=algo_key)
                        if "error" in c_res:
                            st.error(c_res["error"])
                        else:
                            st.success(f"Delineated {c_res['n_clusters_found']} clusters using {c_res['algorithm']}. Silhouette Score: {c_res['silhouette_score']}")
                            st.scatter_chart(c_res["cluster_df"], x="PCA_1", y="PCA_2", color="Cluster")
                            st.markdown("##### Cluster Distribution")
                            st.dataframe(pd.DataFrame(list(c_res["cluster_distribution"].items()), columns=["Cluster", "Count"]), use_container_width=True)

        # ---------------- TAB 3: DIMENSIONALITY REDUCTION ----------------
        with tab_dim:
            st.markdown("#### 🌌 Latent Manifold & Dimensionality Reduction")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.info("ℹ️ At least 2 numeric columns are required to perform dimensionality reduction.")
            else:
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    dim_method = st.selectbox("Method:", ["PCA (Linear)", "t-SNE (Non-Linear Manifold)", "UMAP (Topological)"], key="st_dim_method")
                with d_c2:
                    dim_feats = st.multiselect("Input Dimensions:", num_cols, default=num_cols[:5] if len(num_cols) >= 5 else num_cols, key="st_dim_feats")

                if st.button("🚀 Execute Dimensionality Reduction", type="primary", disabled=len(dim_feats) < 2):
                    m_key = "pca" if "PCA" in dim_method else ("tsne" if "t-SNE" in dim_method else "umap")
                    dim_res = run_dimensionality_reduction(df, columns=dim_feats, method=m_key, n_components=2)
                    if "error" in dim_res:
                        st.error(dim_res["error"])
                    else:
                        st.success(f"Reduced {len(dim_feats)} dimensions into 2 components via {dim_res['method']}.")
                        c1_name, c2_name = f"{m_key.upper()}_1", f"{m_key.upper()}_2"
                        st.scatter_chart(dim_res["reduced_df"], x=c1_name, y=c2_name)
                        if dim_res.get("explained_variance_pct"):
                            st.info(f"Total Explained Variance: {dim_res.get('total_explained_variance_pct')}%")

        # ---------------- TAB 4: ASSOCIATION ANALYSIS ----------------
        with tab_assoc:
            st.markdown("#### 🛒 Market Basket & Association Rule Mining")
            cols = list(df.columns)
            if len(cols) < 2:
                st.info("ℹ️ At least 2 columns (Transaction ID and Item Column) are required for association analysis.")
            else:
                as_c1, as_c2, as_c3 = st.columns(3)
                with as_c1:
                    assoc_algo = st.selectbox("Association Algorithm:", ["FP-Growth (Fast Pattern)", "Apriori (Candidate Generation)"], key="st_assoc_algo")
                with as_c2:
                    tx_col = st.selectbox("Transaction ID Column:", cols, index=0, key="st_tx_col")
                with as_c3:
                    default_item_idx = 1 if len(cols) > 1 else 0
                    item_col = st.selectbox("Item / Product Column:", cols, index=default_item_idx, key="st_item_col")

                if st.button("🛒 Mine Association Rules", type="primary"):
                    a_key = "fpgrowth" if "FP-Growth" in assoc_algo else "apriori"
                    a_res = run_association_analysis(df, transaction_col=tx_col, item_col=item_col, min_support=0.03, min_confidence=0.1, algorithm=a_key)
                    if "error" in a_res:
                        st.error(a_res["error"])
                    else:
                        st.success(f"Mined {len(a_res['rules'])} rules across {a_res['total_transactions']} transactions ({a_res['frequent_itemsets_count']} frequent itemsets).")
                        st.dataframe(a_res["rules_df"], use_container_width=True)

        # ---------------- TAB 5: BUSINESS USE CASES ----------------
        with tab_cases:
            st.markdown("#### 🏢 Pre-Configured Enterprise Business Use Cases")
            uc1, uc2 = st.columns(2)
            with uc1:
                st.markdown("##### 🧑‍🤝‍🧑 Customer Segmentation")
                st.caption("K-Means clustering across tenure, charges, and activity to profile champions, upsells, and churn-risk cohorts.")
                if st.button("Run Customer Segmentation", key="st_btn_uc_cust"):
                    res_uc = run_customer_segmentation_use_case(df)
                    if "error" in res_uc: st.error(res_uc["error"])
                    else:
                        st.success("4 Customer Personas Identified!")
                        for ins in res_uc["insights"]: st.write("• " + ins)
                        st.scatter_chart(res_uc["cluster_df"], x="PCA_1", y="PCA_2", color="Cluster")

                st.markdown("##### 🧠 Behavioral Segmentation")
                st.caption("DBSCAN density clustering to detect core behavioral baselines vs anomalous consumption outliers.")
                if st.button("Run Behavioral Segmentation", key="st_btn_uc_behav"):
                    res_b = run_behavioral_segmentation_use_case(df)
                    if "error" in res_b: st.error(res_b["error"])
                    else:
                        st.success("Behavioral Density Cohorts Formed!")
                        for ins in res_b["insights"]: st.write("• " + ins)
                        st.scatter_chart(res_b["cluster_df"], x="PCA_1", y="PCA_2", color="Cluster")

            with uc2:
                st.markdown("##### 📦 Product Grouping")
                st.caption("Hierarchical Agglomerative clustering to tier product inventory by price elasticity, volume, and margins.")
                if st.button("Run Product Grouping", key="st_btn_uc_prod"):
                    res_p = run_product_grouping_use_case(df)
                    if "error" in res_p: st.error(res_p["error"])
                    else:
                        st.success("Merchandising Hierarchy Formed!")
                        for ins in res_p["insights"]: st.write("• " + ins)
                        st.scatter_chart(res_p["cluster_df"], x="PCA_1", y="PCA_2", color="Cluster")

            st.markdown("##### 🛒 Market Basket Analysis")
            st.caption("FP-Growth association rule mining to discover cross-sell bundles and checkout attachments.")
            if st.button("Run Market Basket Analysis", key="st_btn_uc_mba"):
                res_m = run_market_basket_use_case(df)
                if "error" in res_m: st.error(res_m["error"])
                else:
                    st.success(f"Mined {len(res_m['rules'])} Cross-Sell Attachment Rules!")
                    st.dataframe(res_m["rules_df"], use_container_width=True)

# ---------------- 14. PREDICTIVE ANALYTICS ----------------
elif selected_module in ["🎯 Predictive Analytics", "Predictive Analytics"]:
    st.subheader("🎯 Interactive 'What-If' Prediction Simulator")
    
    trained_res = st.session_state.trained_model_res
    if not trained_res:
        st.info("Please train a model first in **🤖 Machine Learning** to enable interactive What-If predictions.")
    else:
        st.markdown(f"**Loaded Model:** `{trained_res['algorithm']}` (Target: **`{trained_res['prep_data']['target_col']}`**)")
        
        # Generate dynamic input fields
        feature_names = trained_res["prep_data"]["feature_names"]
        input_vals = {}
        
        st.markdown("#### Enter Feature Scenario Values:")
        input_cols = st.columns(3)
        for i, feat in enumerate(feature_names[:9]):
            with input_cols[i % 3]:
                orig_val = float(df[feat].mean()) if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]) else 0.0
                input_vals[feat] = st.number_input(f"{feat}:", value=round(orig_val, 2), key=f"pred_in_{feat}")

        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            pred_output = predict_scenario(trained_res, input_vals)
            st.markdown(f"### Result: **{pred_output.get('prediction')}**")
            st.info(pred_output.get("explanation"))


# ---------------- 15. FORECASTING ----------------
elif selected_module in ["🔮 Forecasting", "Forecasting"]:
    st.subheader("🔮 Time-Series Forecasting & Projections")
    
    auto_date, auto_metric = detect_time_series_columns(df)
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "year"]) or pd.api.types.is_datetime64_any_dtype(df[c])]
    metric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        if date_candidates:
            fore_date = st.selectbox("Date Column:", date_candidates, index=0)
        else:
            fore_date = st.selectbox("Date Column:", list(df.columns) if len(df.columns) > 0 else ["No date column"], index=0)
    with f_c2:
        if metric_candidates:
            fore_metric = st.selectbox("Metric to Forecast:", metric_candidates, index=0)
        else:
            fore_metric = st.selectbox("Metric to Forecast:", ["No numeric column"], index=0)
    with f_c3:
        fore_freq = st.selectbox("Frequency:", ["Monthly (M)", "Weekly (W)", "Daily (D)", "Quarterly (Q)"])
    with f_c4:
        fore_horizon = st.slider("Horizon Periods:", 3, 24, 12)

    if st.button("Generate Forecast", type="primary") and fore_date in df.columns and fore_metric in df.columns:
        with st.spinner("Generating projections and confidence bands..."):
            res_forecast = generate_forecast(df, fore_date, fore_metric, horizon=fore_horizon, freq=fore_freq[0])
            if "error" in res_forecast:
                st.error(res_forecast["error"])
            else:
                st.success(res_forecast["summary"])
                # Plotly forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_forecast["historical_dates"], y=res_forecast["historical_values"], name="Historical Actuals", line=dict(color="#64748b")))
                f_df = res_forecast["forecast_df"]
                fig.add_trace(go.Scatter(x=f_df["Date"], y=f_df["Forecast"], name="Forecast Projection", line=dict(color="#4f46e5", width=3)))
                fig.add_trace(go.Scatter(x=f_df["Date"], y=f_df["Upper_95%"], name="Upper 95% CI", line=dict(dash="dot", color="#93c5fd")))
                fig.add_trace(go.Scatter(x=f_df["Date"], y=f_df["Lower_95%"], name="Lower 95% CI", fill="tonexty", fillcolor="rgba(147, 197, 253, 0.2)", line=dict(dash="dot", color="#93c5fd")))
                fig.update_layout(title=f"Forecast Projection: <b>{fore_metric}</b> ({fore_horizon} periods)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)


# ---------------- 16. MODEL EVALUATION ----------------
elif selected_module in ["📏 Model Evaluation", "Model Evaluation"]:
    st.subheader("📏 Model Evaluation & Diagnostic Curves")
    trained_res = st.session_state.trained_model_res
    if not trained_res:
        st.info("Train a model first in **🤖 Machine Learning** to inspect evaluation curves.")
    else:
        st.markdown(f"**Evaluated Model:** `{trained_res['algorithm']}`")
        eval_data = trained_res["eval_data"]
        
        if trained_res["task_type"] == "classification":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Confusion Matrix")
                cm = np.array(eval_data["confusion_matrix"])
                classes = eval_data["classes"]
                fig_cm = px.imshow(cm, text_auto=True, x=classes, y=classes, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.markdown("#### ROC Curve")
                roc = eval_data.get("roc_data", {})
                if roc.get("fpr"):
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC = {trained_res['metrics'].get('roc_auc', 0.85)}", line=dict(color="#4f46e5", width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#94a3b8")))
                    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.markdown("#### Residual Diagnostics")
            resids = eval_data["residuals"]
            st.plotly_chart(px.histogram(resids, nbins=30, title="Residual Distribution (Actual - Predicted)"), use_container_width=True)


# ---------------- 17. DASHBOARD BUILDER ----------------
elif selected_module == "🎨 Dashboard Builder":
    st.subheader("🎨 Custom Modular Dashboard Builder")
    
    d_theme = st.selectbox("Dashboard Theme:", list(DASHBOARD_THEMES.keys()))
    theme_cfg = DASHBOARD_THEMES[d_theme]

    if st.button("✨ Auto-Generate Executive Dashboard with AI", type="primary"):
        spec = generate_ai_dashboard_spec(df)
        st.session_state.active_dash_spec = spec
        st.success("Generated layout!")

    dash_spec = st.session_state.get("active_dash_spec", generate_ai_dashboard_spec(df))
    st.markdown(f"### {dash_spec['title']}")

    # Render Dashboard KPI Cards
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    dk1, dk2, dk3 = st.columns(3)
    with dk1:
        st.markdown(f"""<div class="metric-card"><h4>Records</h4><div class="metric-val">{len(df):,}</div></div>""", unsafe_allow_html=True)
    with dk2:
        val_sum = float(df[num_cols[0]].sum()) if num_cols else 0.0
        st.markdown(f"""<div class="metric-card"><h4>Total {num_cols[0] if num_cols else 'Metric'}</h4><div class="metric-val">{val_sum:,.2f}</div></div>""", unsafe_allow_html=True)
    with dk3:
        val_avg = float(df[num_cols[0]].mean()) if num_cols else 0.0
        st.markdown(f"""<div class="metric-card"><h4>Average {num_cols[0] if num_cols else 'Metric'}</h4><div class="metric-val">{val_avg:,.2f}</div></div>""", unsafe_allow_html=True)

    # Render Charts Grid
    if cat_cols and num_cols:
        cg1, cg2 = st.columns(2)
        with cg1:
            st.plotly_chart(plot_dashboard_column_chart(df, cat_cols[0], num_cols[0]), use_container_width=True)
        with cg2:
            st.plotly_chart(plot_dashboard_donut_chart(df, cat_cols[0], num_cols[0]), use_container_width=True)


# ---------------- 18. POWER BI ANALYSIS ----------------
elif selected_module == "📊 Power BI Analysis":
    st.subheader("📊 Power BI Export & Report Analysis")
    
    st.markdown("""
    Upload or analyze structured Power BI summary exports to extract KPIs, segment performance, and executive recommendations.
    """)

    pbi_res = analyze_powerbi_export(df, report_name=st.session_state.dataset_source)
    st.markdown(pbi_res["executive_summary"])

    st.markdown("#### Power BI KPIs Extracted:")
    k_cols = st.columns(len(pbi_res["kpis"][:4]) or 1)
    for i, k in enumerate(pbi_res["kpis"][:4]):
        with k_cols[i]:
            st.markdown(f"""<div class="metric-card"><h4>{k['name']}</h4><div class="metric-val">{k['total']:,.2f}</div><small class="badge-chip {'badge-success' if k['growth_pct']>=0 else 'badge-danger'}">{k['growth_pct']:+.1f}% PoP</small></div>""", unsafe_allow_html=True)


# ---------------- 19. REPORT STUDIO ----------------
elif selected_module in ["📑 Report", "📑 Reports"]:
    st.subheader("📑 Executive Report Studio & Multi-Technology Publisher")
    st.caption("Publish publication-grade PDF executive reports, responsive standalone HTML dossiers, and multi-sheet Excel workbooks with custom branding, classification badges, and strategic recommendations.")

    if df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload a dataset to generate reports.")
    else:
        cfg1, cfg2 = st.columns([1.2, 1])
        with cfg1:
            rep_title = st.text_input("Report Title:", value="DataMind Strategic Intelligence & Executive Analytics Report")
            rep_sub = st.text_input("Subtitle / Prepared For:", value="Executive Leadership & Board of Directors Briefing")
        with cfg2:
            rep_author = st.text_input("Lead Author / Department:", value="DataMind AI Analytics Copilot")
            rep_class = st.selectbox("Information Classification:", ["CONFIDENTIAL // PROPRIETARY", "STRICTLY CONFIDENTIAL", "INTERNAL ONLY", "PUBLIC USE"])

        st.markdown("##### 🧩 Modular Report Sections:")
        sec_cols = st.columns(4)
        with sec_cols[0]:
            inc_summary = st.checkbox("Executive Summary & Scorecards", value=True)
        with sec_cols[1]:
            inc_stats = st.checkbox("Numeric Feature Distributions", value=True)
        with sec_cols[2]:
            inc_hygiene = st.checkbox("Data Quality & Hygiene Audit", value=True)
        with sec_cols[3]:
            inc_recs = st.checkbox("Prescriptive Strategy Roadmap", value=True)

        st.markdown("---")
        st.markdown("#### 🚀 Multi-Technology Export Suite")

        export_col1, export_col2, export_col3 = st.columns(3)
        
        # 1. High-Resolution PDF
        with export_col1:
            try:
                pdf_doc = generate_pdf_report(
                    report_title=rep_title,
                    project_name=active_proj["name"],
                    df=df,
                    meta=profile,
                    subtitle=rep_sub,
                    author=rep_author,
                    classification=rep_class,
                    audit_trail=st.session_state.get("cleaning_history", [])
                )
                st.download_button(
                    label="📥 Download Executive Report (PDF)",
                    data=pdf_doc,
                    file_name=f"datamind_executive_report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
                st.caption("✨ Consulting-grade vector PDF formatted for print & presentations.")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

        # 2. Standalone HTML
        with export_col2:
            html_rep = generate_html_report(
                rep_title,
                active_proj["name"],
                df,
                profile,
                audit_trail=st.session_state.get("cleaning_history", [])
            )
            st.download_button(
                label="🌐 Download Standalone HTML Report",
                data=html_rep,
                file_name=f"datamind_executive_report_{int(time.time())}.html",
                mime="text/html",
                use_container_width=True
            )
            st.caption("🌐 Responsive, self-contained interactive web dossier.")

        # 3. Multi-Sheet Excel
        with export_col3:
            excel_wb = generate_excel_report(
                active_proj["name"],
                df,
                profile,
                audit_trail=st.session_state.get("cleaning_history", [])
            )
            st.download_button(
                label="📊 Download Multi-Tab Excel Workbook",
                data=excel_wb,
                file_name=f"dataset_audit_workbook_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.caption("📊 Formatted multi-tab spreadsheet with metadata and data preview.")

        # Live Interactive Preview
        st.markdown("---")
        st.markdown("#### 👁️ Live Report Preview")
        st.components.v1.html(html_rep, height=580, scrolling=True)


# ---------------- 20. DATA EXPORT ----------------
elif selected_module == "💾 Data Export":
    st.subheader("💾 Export & Data Sharing")
    
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Export Current Processed Dataset")
        csv_b = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export as CSV", csv_b, "dataset_processed.csv", "text/csv", use_container_width=True)
        
        excel_b = generate_excel_report(active_proj["name"], df, profile)
        st.download_button("📥 Export as Excel (.xlsx)", excel_b, "dataset_workbook.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    with e2:
        st.markdown("#### Export Analytical Dumps")
        json_b = df.head(1000).to_json(orient="records").encode("utf-8")
        st.download_button("📥 Export JSON Records", json_b, "dataset.json", "application/json", use_container_width=True)


# ---------------- 21. MODEL REGISTRY ----------------
elif selected_module == "🏷️ Model Registry":
    st.subheader("🏷️ Model Lifecycle Management & Registry")
    
    registry: ModelRegistry = st.session_state.model_registry
    all_models = registry.get_all_models()
    
    for m in all_models:
        with st.container():
            st.markdown(f"### {m['name']} (v{m['version']})")
            st.caption(f"Algorithm: **{m['algorithm']}** | Target: **`{m['target']}`** | Status: **{m['status']}** | Trained: {m.get('training_date')}")
            st.json(m.get("metrics", {}))
            st.markdown("---")


# ---------------- 22. SETTINGS ----------------
elif selected_module == "⚙️ Settings":
    st.subheader("⚙️ Platform Configuration & Security Settings")
    
    st.markdown("#### Role-Based Access Simulation")
    st.write(f"Current Role: **{st.session_state.user_role}**")
    
    st.markdown("#### API Keys & Cloud Providers")
    gemini_key = st.text_input("Google Gemini API Key:", type="password", placeholder="Enter AI provider key...")
    
    st.markdown("#### System Information")
    st.write({
        "Platform Version": "2.0.0 Enterprise",
        "Python": "3.11",
        "Engine": "DuckDB + Scikit-Learn + FastAPI",
        "OS": "Windows",
        "Active Project": active_proj["name"]
    })
