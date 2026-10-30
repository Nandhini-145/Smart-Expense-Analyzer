"""
Script to generate notebooks/analysis.ipynb with clean cells and documentation.
"""
import json
import os

def generate_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 💳 Smart Expense & Spending Analyzer\n",
                    "### End-to-End AI/ML Financial Analytics, Anomaly Detection & Expense Forecasting\n",
                    "**Author:** AI & Data Science Project  \n",
                    "**Tech Stack:** Python, Pandas, Matplotlib, Scikit-Learn, PostgreSQL, Power BI, Streamlit\n",
                    "\n",
                    "---\n",
                    "### Project Workflow:\n",
                    "1. **Data Cleaning & Feature Engineering**\n",
                    "2. **Exploratory Data Analysis (EDA) & Spending Patterns**\n",
                    "3. **AI Expense Categorization (TF-IDF + Logistic Regression)**\n",
                    "4. **AI Anomaly Detection (Isolation Forest)**\n",
                    "5. **Future Expense Forecasting (Linear Regression)**\n",
                    "6. **Smart Financial Insights & Budget Recommendation Engine**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Library Imports\n",
                    "import os\n",
                    "import sys\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set chart style\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "print('Libraries imported successfully.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Cleaning & Feature Engineering"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sys.path.append('..')\n",
                    "from src.preprocessing import load_and_preprocess_data\n",
                    "\n",
                    "data_path = '../data/expenses.csv' if os.path.exists('../data/expenses.csv') else 'data/expenses.csv'\n",
                    "df = load_and_preprocess_data(data_path)\n",
                    "\n",
                    "print(f'Total Cleaned Transactions: {len(df)}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Spending Analysis & Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.spending_analysis import compute_spending_summary, plot_spending_analysis\n",
                    "\n",
                    "summary = compute_spending_summary(df)\n",
                    "print('Total Expenses :', f'₹{summary[\"total_expense\"]:,.2f}')\n",
                    "print('Average Expense:', f'₹{summary[\"average_expense\"]:,.2f}')\n",
                    "print('Highest Category:', summary['highest_spending_category'])\n",
                    "print('Lowest Category :', summary['lowest_spending_category'])"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Category Spending Distribution\n",
                    "cat_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.barplot(x=cat_totals.values, y=cat_totals.index, palette='Blues_r')\n",
                    "plt.title('Total Spending by Category (₹)', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Spending Amount (₹)')\n",
                    "plt.ylabel('Category')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. AI Expense Category Prediction (TF-IDF + Logistic Regression)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.categorization import train_category_model, predict_category\n",
                    "\n",
                    "pipeline, acc = train_category_model(data_path=data_path)\n",
                    "\n",
                    "# Test interactive inference\n",
                    "test_items = ['Swiggy dinner order', 'Amazon Prime video', 'Shell petrol bunk fuel', 'Apollo medicines']\n",
                    "for item in test_items:\n",
                    "    cat, conf = predict_category(item, model=pipeline)\n",
                    "    print(f'\"{item}\" -> Predicted Category: [{cat}] (Confidence: {conf*100:.1f}%)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. AI Anomaly Detection (Isolation Forest)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.anomaly import train_anomaly_model, detect_anomalies\n",
                    "\n",
                    "iso_model, df_anom = train_anomaly_model(data_path=data_path)\n",
                    "anomalies = df_anom[df_anom['is_anomaly'] == 1]\n",
                    "print(f'Detected {len(anomalies)} anomalies out of {len(df_anom)} total transactions.')\n",
                    "anomalies[['date_str', 'description', 'category', 'amount', 'payment_method']].head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Future Expense Forecasting (Linear Regression)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.forecasting import train_forecast_model, forecast_next_months\n",
                    "\n",
                    "model, monthly_df, next_pred, metrics = train_forecast_model(data_path=data_path)\n",
                    "print(f'Projected Next Month Spending: ₹{next_pred:,.2f}')\n",
                    "\n",
                    "# Multi-month projection\n",
                    "fut_df = forecast_next_months(df, n_months=3)\n",
                    "fut_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. AI Insights & Smart Budget Recommendation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.insights import generate_spending_insights, evaluate_smart_budget\n",
                    "\n",
                    "print('--- AI Spending Insights ---')\n",
                    "for ins in generate_spending_insights(df):\n",
                    "    print(ins)\n",
                    "\n",
                    "print('\\n--- Smart Budget Tracker ---')\n",
                    "budget_res = evaluate_smart_budget(monthly_spending=27500, monthly_budget=25000)\n",
                    "print(budget_res)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Successfully generated notebooks/analysis.ipynb")

if __name__ == "__main__":
    generate_notebook()
