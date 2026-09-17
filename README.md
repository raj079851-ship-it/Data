# DataMind AI | Next-Gen AI-Powered Data Analytics Platform

A full-stack, interactive AI Data Analytics platform built with Python, Streamlit, Pandas, Scikit-Learn, Scipy, and Plotly. Upload any CSV file or pick from built-in sample datasets to perform end-to-end data analysis across three core pillars: **Data Cleaning**, **Feature Engineering**, and **Exploratory Data Analysis (EDA)**.

---

## 🚀 Quick Start

### 1. Launch with One Click (Windows)
Double-click `run.bat` in this folder.

### 2. Manual Terminal Launch
```powershell
python -m streamlit run app.py
```
Or with specific Python path:
```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

Then open your browser at **http://localhost:8501**.

---

## 🌟 Key Features

### 1. 📂 Data Ingestion & Profiling
- **CSV Drag-and-Drop**: Upload custom CSVs with automated or manual delimiter (`,`, `;`, `\t`, `|`) and encoding detection (`utf-8`, `latin1`, `cp1252`).
- **Built-in Demo Datasets**: Instant access to **Telecom Customer Churn** and **Real Estate Housing** datasets (pre-configured with realistic missing values, outliers, and dirty data for immediate testing).
- **Top Metrics Bar**: Live KPI cards showing dataset dimensions, overall Data Health Score (0-100), missing cells count/%, duplicates, and memory usage.

### 2. 🧹 Data Cleaning Module
- **Comprehensive Quality Audit**: Automatically checks missing rates, duplicate records, zero-variance constant columns, and numeric outliers.
- **🚀 1-Click AI Auto-Clean**: In one click, deduplicates records, normalizes string whitespace, drops uninformative columns, imputes missing numbers with Median and categoricals with Mode, and caps extreme outliers using IQR.
- **Granular Cleaning Sub-tabs**:
  - **Missing Value Imputation**: Impute via Mean, Median, Mode, Constant, Forward/Backward Fill, or row/column drops.
  - **Outlier Handling**: Detect via IQR (1.5x) or Z-score (>3.0) and choose between Capping (Winsorization), Trimming (dropping rows), or setting to NaN.
  - **Duplicates & Text**: Deduplicate rows, trim whitespace, and normalize case (`lower`, `upper`, `title`).
  - **Type Casting**: Cast columns to numeric, datetime, categorical, boolean, or string.
- **Cleaning Changelog & CSV Export**: Download the cleaned dataset anytime.

### 3. ⚙️ Feature Engineering Module
- **Smart Column Categorization**: Automatically categorizes columns into numerical, low-cardinality categorical, high-cardinality, datetime, and text/IDs.
- **AI Feature Suggestions**: Proactive recommendations with rationale (e.g. log transform on skewed features, date decomposition, interaction ratios).
- **Transformation Tools**:
  - **Scaling & Normalization**: Standard Scaler, MinMax Scaler, and Robust Scaler.
  - **Non-Linear & Binning**: Log1p, Square Root, and Quantile/Uniform Binning.
  - **Categorical Encoding**: One-Hot Encoding (with top-N category control), Label Encoding, and Frequency Encoding.
  - **Datetime Extraction**: Decompose timestamps into Year, Month, Day, Day of Week, Is_Weekend, and Quarter.
  - **Interaction & Ratios**: Synthesize ratios, products, sums, and differences between any two numerical columns.

### 4. 📊 Exploratory Data Analysis (EDA) Module
- **Summary Statistics**: Central tendency, dispersion, IQR, skewness, and kurtosis for numeric features; counts, unique levels, and prevalence for categoricals.
- **Univariate Distributions**: Interactive Plotly histograms with marginal box plots and categorical frequency bar charts.
- **Correlation Heatmap**: Annotated Pearson heatmap with automatic high collinearity warnings (|r| ≥ 0.80) and top positive/negative pairs.
- **Bivariate & Multivariate Exploration**:
  - Interactive scatter plots with color dimensions and OLS trendlines.
  - Category-grouped box plots for distribution comparisons.
  - Multivariate pairwise scatter matrices (pairplots).
- **Target Variable Deep-Dive**: Select any outcome column to automatically rank predictor features by Pearson correlation or ANOVA F-Score.

### 5. 📈 Interactive BI Dashboard Module (NEW!)
- **Dynamic Real-Time Slicers**: Filter by Category Segment, select Primary Metric, Group-by Dimension, and choose Aggregation method (`Sum`, `Average`, `Count`, `Max`, `Min`).
- **Executive Metric Cards**: Filtered records count, aggregate metric sum, average per record, and metric range.
- **Multi-Chart Grid**:
  - Aggregated Segment Bar Breakdown with dynamic data labels.
  - Segment Composition Donut / Pie Chart.
  - Comparative Box Plot across categories.
  - Top Performers Leaderboard with medal ranks (🥇, 🥈, 🥉).
- **Interactive Pivot / Cross-Tab Builder**: Generate multi-dimensional aggregated matrices across any two attributes.

### 6. 🧠 AI Insights & Executive Report
- **Built-in Statistical AI Engine**: Generates an executive narrative, data hygiene alerts, anomaly flags, and strategic modeling recommendations offline with zero API key requirement.
- **Optional Gemini Assistant**: Enter an optional Google Gemini API key to ask natural language questions directly about your dataset.

---

## 🧪 Running Automated Tests

Run the full unit and integration test suite:
```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" tests/test_modules.py
```
All tests verify data loading, quality auditing, missing value imputation, outlier handling, feature scaling, encoding, datetime extraction, EDA calculations, and AI insights generation.
