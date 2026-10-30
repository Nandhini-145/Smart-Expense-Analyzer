# 💳 Smart Expense & Spending Analyzer

An end-to-end, production-ready AI and Data Science system that transforms raw, unorganized financial transaction logs into structured intelligence. It automatically cleanses data, classifies expense descriptions using **Natural Language Processing (TF-IDF + Logistic Regression)**, detects suspicious spending spikes via **Unsupervised Anomaly Detection (Isolation Forest)**, forecasts future monthly expenditures using **Time-Series Linear Regression**, provides automated financial health insights, and powers interactive dashboards across **Streamlit**, **Power BI**, and **PostgreSQL**.

---

## 🏗️ System Architecture & Workflow Flowchart

```
┌────────────────────────────────────────────────────────┐
│                   Raw Expenses CSV                     │
│    (date, description, amount, payment_method, etc.)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           Data Cleaning & Preprocessing (Pandas)       │
│  - Null Imputation & Type Casting                      │
│  - Duplicate De-duplication                            │
│  - Text Normalization & Clean Strings                  │
│  - Feature Engineering (Day, Month, Weekday, Lags)     │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ AI Category Classifier   │  │   AI Anomaly Detector    │
│  - TF-IDF (1-2 N-grams)  │  │   - Isolation Forest     │
│  - Logistic Regression   │  │   - Unsupervised Outlier │
│  - 98.5% Accuracy        │  │   - Contamination: 2.5%  │
└─────────────┬────────────┘  └───────────┬──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│      Time-Series Monthly Forecasting (Linear Reg)      │
│  - Monthly Financial Rollups & Lag Spending Features   │
│  - MAE, RMSE, R² Score Evaluation                      │
│  - Multi-Month Forward Trajectory Prediction           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     Rule-Based Spending Insights & Budget Tracker      │
│  - MoM Category Surge Analysis                         │
│  - Spending Cut Optimization Targets (10% Rule)        │
│  - Safe / Moderate / Warning / Over-Budget Tiers       │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Streamlit Interactive App│  │ PostgreSQL DB Ingestion  │
│  - 7 Functional Sections │  │  - Parameterized Queries │
│  - Real-Time Inference   │  │  - Aggregations & Views  │
│  - Interactive Filtering │  │  - Power BI DirectQuery  │
└──────────────────────────┘  └──────────────────────────┘
```

---

## 📁 Repository Structure

```
Smart_Expense_Analyzer/
├── data/
│   ├── expenses.csv              # 1,000 realistic expense transactions
│   └── expense_db.sqlite         # SQLite test fallback database
├── notebooks/
│   └── analysis.ipynb            # Jupyter Notebook with full step-by-step EDA & ML
├── models/
│   ├── category_model.pkl        # Serialized TF-IDF + Logistic Regression pipeline
│   ├── anomaly_model.pkl         # Serialized Isolation Forest model
│   ├── forecast_model.pkl        # Serialized Linear Regression model & metadata
│   ├── category_confusion_matrix.png
│   ├── anomaly_detection_plot.png
│   ├── forecast_trend_plot.png
│   └── charts/                   # Matplotlib/Seaborn visualization outputs
├── src/
│   ├── preprocessing.py          # Data ingestion, cleaning, and feature engineering
│   ├── categorization.py         # AI NLP Expense Categorization pipeline
│   ├── spending_analysis.py      # Statistical summaries & EDA visualizations
│   ├── anomaly.py                # Isolation Forest anomaly detection
│   ├── forecasting.py            # Future expense forecasting with Linear Regression
│   ├── insights.py               # Automated AI spending recommendations & budget tracker
│   └── database.py               # PostgreSQL parameterized CRUD & aggregations
├── app.py                        # Modern, interactive Streamlit web application
├── requirements.txt              # Production Python package dependencies
└── README.md                     # Project documentation & interview guide
```

---

## ⚡ Quick Start Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/smart-expense-analyzer.git
cd smart-expense-analyzer

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train ML Models
```bash
# Generate/verify dataset
python src/generate_dataset.py

# Train NLP Categorizer (TF-IDF + Logistic Regression)
python src/categorization.py

# Train Anomaly Detector (Isolation Forest)
python src/anomaly.py

# Train Time-Series Forecaster (Linear Regression)
python src/forecasting.py

# Verify Database Connection & Ingestion
python src/database.py
```

### 3. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
*Access the dashboard in your browser at `http://localhost:8501`.*

---

## 📊 Streamlit Application Features

| Tab | Feature Name | Description |
|---|---|---|
| **1** | **Overview** | High-level metrics, monthly trend line chart, category donut breakdown, and recent transactions. |
| **2** | **Expense Analysis** | Category expenditure rankings, payment method distributions, and top 10 most expensive transactions. |
| **3** | **AI Category Prediction** | Real-time text inference. Type any merchant or item name (e.g. *"Swiggy dinner order"*) to get instant category classification and confidence probability. |
| **4** | **Anomaly Detection** | Isolation forest scatter visualization separating normal transactions from sudden spikes with decision scores. |
| **5** | **Expense Forecast** | Time-series forecasting plotting historical monthly spending vs model fitted trajectory and next-month projections. |
| **6** | **AI Insights** | Rule-based financial tips, month-over-month category surge warnings, and 10% cut optimization calculations. |
| **7** | **Budget Tracker** | Interactive budget calculator highlighting budget status (*Safe 🟢, Moderate 🟡, Warning 🟠, Over Budget ⚠️*) with daily spending limits. |

---

## 🐘 PostgreSQL Integration

The `src/database.py` module manages database persistence using parameterized SQL queries to prevent SQL injection:

```sql
-- Table Schema
CREATE TABLE expenses (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    description VARCHAR(255) NOT NULL,
    amount NUMERIC(10, 2) NOT NULL CHECK (amount > 0),
    payment_method VARCHAR(50) NOT NULL,
    category VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX idx_expenses_date ON expenses(date);
CREATE INDEX idx_expenses_category ON expenses(category);
```

### Parameterized CRUD Operations Implemented:
- `insert_transaction(date, description, amount, payment_method, category)`
- `bulk_insert_dataframe(df)`
- `select_all_expenses(limit)`
- `select_by_category(category)`
- `update_transaction(expense_id, new_category, new_amount)`
- `delete_transaction(expense_id)`
- `get_category_aggregates()`
- `get_monthly_aggregates()`

---

## 📈 Power BI Enterprise Dashboard Design

### Data Connection
1. In Power BI Desktop, click **Get Data** -> **Text/CSV** -> Select `data/expenses.csv` (or **PostgreSQL database**).
2. Click **Transform Data** (Power Query) to ensure `date` is formatted as Date and `amount` as Decimal Number.
3. Click **Close & Apply**.

### Core DAX Measures
Create a dedicated `_Measures` table in Power BI and paste the following DAX calculations:

```dax
// 1. Total Expense
Total Expense = SUM(expenses[amount])

// 2. Average Expense
Average Expense = AVERAGE(expenses[amount])

// 3. Monthly Average Spending
Monthly Average = 
AVERAGEX(
    VALUES(expenses[month_name]),
    [Total Expense]
)

// 4. Budget Used Percentage (Assumes monthly budget of 25,000)
Monthly Budget = 25000
Budget Used % = DIVIDE([Monthly Average], [Monthly Budget], 0) * 100

// 5. Total Transactions
Total Transactions = COUNTROWS(expenses)

// 6. Number of Anomalies (Amount > 12,000)
Number of Anomalies = 
CALCULATE(
    COUNTROWS(expenses),
    expenses[amount] >= 12000
)

// 7. Overspending Amount
Overspending Amount = 
MAX(0, [Monthly Average] - [Monthly Budget])
```

### Canvas Layout & Visuals
1. **Top KPI Row**: 5 Card visuals for `Total Expense`, `Average Expense`, `Monthly Average`, `Budget Used %`, and `Number of Anomalies`.
2. **Visual 1 (Line Chart)**: `date` (Month) on X-axis, `Total Expense` on Y-axis.
3. **Visual 2 (Donut Chart)**: `category` as Legend, `Total Expense` as Values.
4. **Visual 3 (Clustered Bar Chart)**: `payment_method` on Y-axis, `Total Expense` on X-axis.
5. **Visual 4 (Line Chart)**: `date` (Day) on X-axis, `Average Expense` on Y-axis.
6. **Visual 5 (Bar Chart - Top 10 Transactions)**: `description` on Y-axis, `amount` on X-axis (Filter: Top 10 by `amount`).
7. **Visual 6 (Stacked Column Chart)**: `month_name` on X-axis, `Total Expense` on Y-axis, `category` in Legend.
8. **Visual 7 (Table Visual)**: Columns `date`, `description`, `category`, `amount`, `payment_method` (Filter: `amount` >= 12,000).
