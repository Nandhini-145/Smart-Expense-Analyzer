"""
Smart Expense & Spending Analyzer - Streamlit Application
File: app.py
Author: Antigravity AI
Description: A modern, enterprise-ready dashboard for intelligent expense tracking,
AI categorization, Isolation Forest anomaly detection, Linear Regression forecasting,
rule-based financial insights, and smart budget tracking.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modular project components
from src.preprocessing import load_and_preprocess_data, clean_text
from src.categorization import load_category_model, predict_category
from src.spending_analysis import compute_spending_summary
from src.anomaly import detect_anomalies, check_single_transaction_anomaly
from src.forecasting import prepare_monthly_dataset, train_forecast_model, forecast_next_months
from src.insights import generate_spending_insights, evaluate_smart_budget

# Page configuration
st.set_page_config(
    page_title="Smart Expense & Spending Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .kpi-card {
        background: white;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.07);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/wallet.png", width=70)
st.sidebar.title("Analyzer Settings")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Expense CSV",
    type=["csv"],
    help="Upload your bank/expense statement CSV with columns: date, description, amount, payment_method, category"
)

# Load data safely
data_source = uploaded_file if uploaded_file is not None else "data/expenses.csv"
try:
    raw_df = load_and_preprocess_data(data_source)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# 2. Date Range Filter
min_date = raw_df["date"].min().date()
max_date = raw_df["date"].max().date()
selected_dates = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 3. Category Filter
all_categories = sorted(raw_df["category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "🏷️ Filter Categories",
    options=all_categories,
    default=all_categories
)

# 4. Payment Method Filter
all_payment_methods = sorted(raw_df["payment_method"].unique().tolist())
selected_pm = st.sidebar.multiselect(
    "💳 Payment Methods",
    options=all_payment_methods,
    default=all_payment_methods
)

# 5. Monthly Budget Input
user_budget = st.sidebar.number_input(
    "🎯 Monthly Budget (₹)",
    min_value=5000,
    max_value=500000,
    value=25000,
    step=2000,
    help="Enter your target monthly spending budget"
)

# Apply filters
if len(selected_dates) == 2:
    start_dt, end_dt = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    filtered_df = raw_df[
        (raw_df["date"] >= start_dt) &
        (raw_df["date"] <= end_dt) &
        (raw_df["category"].isin(selected_categories)) &
        (raw_df["payment_method"].isin(selected_pm))
    ].copy()
else:
    filtered_df = raw_df[
        (raw_df["category"].isin(selected_categories)) &
        (raw_df["payment_method"].isin(selected_pm))
    ].copy()

if filtered_df.empty:
    st.warning("⚠️ No transactions match the selected filters. Please adjust the sidebar controls.")
    st.stop()

# Compute high-level metrics
summary = compute_spending_summary(filtered_df)
df_anomalies = detect_anomalies(filtered_df)
total_anomalies = int(df_anomalies["is_anomaly"].sum())

# Monthly average spending
monthly_spending_series = filtered_df.groupby(["year", "month"])["amount"].sum()
avg_monthly_spending = float(monthly_spending_series.mean()) if not monthly_spending_series.empty else float(summary["total_expense"])
budget_used_ratio = (avg_monthly_spending / user_budget) * 100

# -------------------------------------------------------------
# MAIN HEADER & KPI CARDS
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.1rem; font-weight:800;">💳 Smart Expense & Spending Analyzer</h1>
    <p style="margin:4px 0 0 0; color:#94A3B8; font-size:1.0rem;">
        AI-Powered Expense Classification, Anomaly Detection & Machine Learning Financial Forecaster
    </p>
</div>
""", unsafe_allow_html=True)

# 6 Dashboard KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Spending</div>
        <div class="kpi-value">₹{summary['total_expense']:,.0f}</div>
        <div class="kpi-sub">{summary['total_transactions']} transactions</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Expense</div>
        <div class="kpi-value">₹{summary['average_expense']:,.0f}</div>
        <div class="kpi-sub">Median: ₹{summary['median_expense']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Monthly Average</div>
        <div class="kpi-value">₹{avg_monthly_spending:,.0f}</div>
        <div class="kpi-sub">Across {len(monthly_spending_series)} months</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    budget_color = "#10B981" if budget_used_ratio <= 70 else ("#F59E0B" if budget_used_ratio <= 90 else "#EF4444")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Budget Used %</div>
        <div class="kpi-value" style="color: {budget_color};">{budget_used_ratio:.1f}%</div>
        <div class="kpi-sub">Target: ₹{user_budget:,.0f}/mo</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Highest Category</div>
        <div class="kpi-value" style="font-size: 1.35rem;">{summary['highest_spending_category']}</div>
        <div class="kpi-sub">₹{summary['highest_spending_amount']:,.0f} spent</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Anomalies Detected</div>
        <div class="kpi-value" style="color: #EF4444;">{total_anomalies}</div>
        <div class="kpi-sub">{(total_anomalies/len(filtered_df))*100:.1f}% of total txs</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 7 INTERACTIVE SECTIONS (TABS)
# -------------------------------------------------------------
tab_overview, tab_analysis, tab_ai_cat, tab_anomaly, tab_forecast, tab_insights, tab_budget = st.tabs([
    "📊 1. Overview",
    "📈 2. Expense Analysis",
    "🤖 3. AI Category Prediction",
    "🚨 4. Anomaly Detection",
    "🔮 5. Expense Forecast",
    "🧠 6. AI Insights",
    "🎯 7. Budget Tracker"
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    st.subheader("Financial Overview & Category Composition")
    col_ov1, col_ov2 = st.columns([3, 2])
    
    with col_ov1:
        # Monthly spending trend
        monthly = filtered_df.groupby(["year", "month", "month_name"])["amount"].sum().reset_index()
        monthly["label"] = monthly["month_name"] + " " + monthly["year"].astype(str)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly["label"], monthly["amount"], marker="o", color="#2563EB", linewidth=2.5, markersize=6)
        ax.fill_between(monthly["label"], monthly["amount"], alpha=0.15, color="#2563EB")
        ax.set_title("Total Monthly Spending Trend", fontsize=11, fontweight="bold")
        ax.set_ylabel("Spending (₹)")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col_ov2:
        # Category Donut Chart
        cat_sums = filtered_df.groupby("category")["amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = sns.color_palette("Set2", len(cat_sums))
        wedges, texts, autotexts = ax.pie(
            cat_sums, labels=cat_sums.index, autopct="%1.1f%%",
            startangle=140, colors=colors, pctdistance=0.8,
            textprops=dict(fontsize=8)
        )
        centre_circle = plt.Circle((0, 0), 0.60, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title("Spending by Category", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Recent Transactions Table
    st.markdown("##### Recent Transactions")
    st.dataframe(
        filtered_df[["date_str", "description", "category", "amount", "payment_method"]].tail(10),
        use_container_width=True
    )

# -------------------------------------------------------------
# TAB 2: EXPENSE ANALYSIS
# -------------------------------------------------------------
with tab_analysis:
    st.subheader("Granular Spending Patterns & Analytics")
    
    col_ea1, col_ea2 = st.columns(2)
    with col_ea1:
        # Category totals bar chart
        cat_bar = filtered_df.groupby("category")["amount"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(cat_bar.index, cat_bar.values, color="#3B82F6", height=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + (cat_bar.max() * 0.01), bar.get_y() + bar.get_height()/2,
                    f"₹{w:,.0f}", va="center", fontsize=8, fontweight="bold")
        ax.set_title("Category-wise Total Expenditure", fontsize=11, fontweight="bold")
        ax.set_xlabel("Spending (₹)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col_ea2:
        # Payment Method Breakdown
        pm_bar = filtered_df.groupby("payment_method")["amount"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(pm_bar.index, pm_bar.values, color="#10B981", height=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + (pm_bar.max() * 0.01), bar.get_y() + bar.get_height()/2,
                    f"₹{w:,.0f}", va="center", fontsize=8, fontweight="bold")
        ax.set_title("Spending Breakdown by Payment Method", fontsize=11, fontweight="bold")
        ax.set_xlabel("Spending (₹)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Top 10 Expensive Transactions
    st.markdown("##### 🏆 Top 10 Highest Spending Transactions")
    top10 = filtered_df.nlargest(10, "amount")[["date_str", "description", "category", "amount", "payment_method"]]
    top10["amount"] = top10["amount"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: AI CATEGORY PREDICTION
# -------------------------------------------------------------
with tab_ai_cat:
    st.subheader("AI Expense Category Prediction (TF-IDF + Logistic Regression)")
    st.markdown("""
    Enter any real-world merchant or expense description. The trained Machine Learning NLP model
    will clean the text, extract TF-IDF unigram & bigram features, and predict the exact financial category.
    """)
    
    col_input, col_pred = st.columns([3, 2])
    with col_input:
        user_desc = st.text_input(
            "Transaction Description",
            value="Swiggy dinner order",
            placeholder="e.g. Netflix subscription, Petrol bunk fuel, Apollo pharmacy..."
        )
        predict_button = st.button("🚀 Predict Category", type="primary")
        
        st.markdown("##### Quick Test Examples:")
        sample_chips = [
            "Amazon laptop wireless mouse",
            "Petrol bunk HP fuel refill",
            "Electricity bill Bescom",
            "Apollo Pharmacy medicines",
            "Udemy Python data science course",
            "Zomato biryani delivery",
            "Zara denim jacket shopping"
        ]
        chip_col1, chip_col2 = st.columns(2)
        for i, chip in enumerate(sample_chips):
            target_col = chip_col1 if i % 2 == 0 else chip_col2
            if target_col.button(chip, key=f"chip_{i}"):
                user_desc = chip
                
    with col_pred:
        cat_model = load_category_model()
        pred_cat, conf = predict_category(user_desc, model=cat_model)
        
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:20px; border-radius:12px; margin-top:28px;">
            <div style="color:#64748B; font-size:0.85rem; font-weight:600; text-transform:uppercase;">Predicted Category</div>
            <div style="color:#2563EB; font-size:2.0rem; font-weight:800; margin-top:6px;">{pred_cat}</div>
            <div style="margin-top:10px; color:#475569; font-size:0.95rem;">
                Confidence Score: <b>{conf*100:.1f}%</b>
            </div>
            <div style="margin-top:12px; background:#E2E8F0; border-radius:8px; height:8px; width:100%; overflow:hidden;">
                <div style="background:#2563EB; height:100%; width:{conf*100:.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("##### 🧠 Machine Learning Architecture:")
    st.code("Input Text -> Text Cleaning -> TF-IDF Vectorizer (1-2 N-grams) -> Logistic Regression Classifier -> Predicted Category", language="text")

# -------------------------------------------------------------
# TAB 4: ANOMALY DETECTION
# -------------------------------------------------------------
with tab_anomaly:
    st.subheader("AI Anomaly Detection (Isolation Forest)")
    st.markdown("""
    Isolation Forest is an unsupervised tree-based algorithm that isolates unexpected spending spikes
    by constructing random partitioning trees. Rare and extreme expenses require fewer splits to isolate.
    """)
    
    anom_k1, anom_k2, anom_k3 = st.columns(3)
    normal_count = len(df_anomalies) - total_anomalies
    anom_k1.metric("Normal Transactions", f"{normal_count:,}")
    anom_k2.metric("Unusual Anomalies Flagged", f"{total_anomalies:,}")
    anom_k3.metric("Anomaly Rate", f"{(total_anomalies/len(df_anomalies))*100:.2f}%")
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(
        df_anomalies[df_anomalies["is_anomaly"] == 0].index,
        df_anomalies[df_anomalies["is_anomaly"] == 0]["amount"],
        c="#3B82F6", label="Normal Transactions", alpha=0.6, s=25
    )
    ax.scatter(
        df_anomalies[df_anomalies["is_anomaly"] == 1].index,
        df_anomalies[df_anomalies["is_anomaly"] == 1]["amount"],
        c="#EF4444", label="Detected Anomalies", alpha=0.9, s=65, marker="X"
    )
    ax.set_title(f"Isolation Forest Anomaly Visualization ({total_anomalies} Outliers Flagged)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Expense Amount (₹)")
    ax.legend(frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("##### 🚨 Flagged Anomalous Transactions Table")
    flagged = df_anomalies[df_anomalies["is_anomaly"] == 1][["date_str", "description", "category", "amount", "payment_method", "anomaly_score"]]
    flagged = flagged.sort_values(by="amount", ascending=False).reset_index(drop=True)
    flagged["amount"] = flagged["amount"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(flagged, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: EXPENSE FORECAST
# -------------------------------------------------------------
with tab_forecast:
    st.subheader("Future Expense Forecasting (Linear Regression)")
    st.markdown("""
    Aggregates your historical monthly spending trajectory, rolling statistics, and transaction volume
    to forecast upcoming monthly commitments using Scikit-Learn's Linear Regression model.
    """)
    
    # Run forecasting
    try:
        model, monthly_df, next_pred, metrics = train_forecast_model()
        future_proj = forecast_next_months(filtered_df, n_months=3)
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Next Month Forecast", f"₹{next_pred:,.0f}")
        fc2.metric("Mean Absolute Error (MAE)", f"₹{metrics['mae']:,.0f}")
        fc3.metric("Root Mean Squared Error", f"₹{metrics['rmse']:,.0f}")
        fc4.metric("R² Score", f"{metrics['r2']:.2f}")
        
        # Forecast Chart
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(monthly_df["month_label"], monthly_df["total_amount"],
                marker="o", label="Actual Monthly Spending", color="#2563EB", linewidth=2.5)
        ax.plot(monthly_df["month_label"], monthly_df["predicted_amount"],
                marker="s", linestyle="--", label="Model Fitted Spending", color="#10B981", linewidth=2)
        
        next_label = f"Month {monthly_df['month_idx'].iloc[-1]+1} (Forecast)"
        ax.plot([monthly_df["month_label"].iloc[-1], next_label],
                [monthly_df["total_amount"].iloc[-1], next_pred],
                marker="*", markersize=12, linestyle=":", color="#EF4444", label=f"Next Month Forecast: ₹{next_pred:,.0f}")
        
        ax.set_title("Historical Spending vs Next-Month AI Forecast", fontsize=11, fontweight="bold")
        ax.set_ylabel("Monthly Total Spending (₹)")
        plt.xticks(rotation=25, ha="right", fontsize=9)
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("##### 📅 Multi-Month Forward Projections:")
        future_proj["predicted_spending"] = future_proj["predicted_spending"].apply(lambda x: f"₹{x:,.2f}")
        st.table(future_proj)
    except Exception as e:
        st.error(f"Error computing forecast: {e}")

# -------------------------------------------------------------
# TAB 6: AI INSIGHTS
# -------------------------------------------------------------
with tab_insights:
    st.subheader("Automated AI Spending Insights & Recommendations")
    st.markdown("Smart financial intelligence extracted dynamically from your spending habits:")
    
    insights = generate_spending_insights(filtered_df)
    for ins in insights:
        st.info(ins)

# -------------------------------------------------------------
# TAB 7: BUDGET TRACKER
# -------------------------------------------------------------
with tab_budget:
    st.subheader("Smart Budget Tracker & Recommendation")
    
    # Evaluate budget
    budget_eval = evaluate_smart_budget(avg_monthly_spending, user_budget)
    
    b_col1, b_col2 = st.columns([1, 1])
    
    with b_col1:
        st.markdown(f"""
        <div style="background:white; border:1px solid #CBD5E1; padding:24px; border-radius:14px; box-shadow:0 2px 10px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.1rem; font-weight:700; color:#1E293B;">Budget Status</span>
                <span class="status-badge" style="background:{budget_eval['badge_color']}; color:white;">
                    {budget_eval['icon']} {budget_eval['status']}
                </span>
            </div>
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#64748B;">
                    <span>Monthly Budget:</span>
                    <b>₹{budget_eval['budget']:,.2f}</b>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#64748B; margin-top:8px;">
                    <span>Actual Average Spending:</span>
                    <b style="color:#0F172A;">₹{budget_eval['actual_spending']:,.2f}</b>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#64748B; margin-top:8px;">
                    <span>Remaining Budget:</span>
                    <b style="color:{'#10B981' if budget_eval['remaining_budget'] >= 0 else '#EF4444'};">
                        ₹{budget_eval['remaining_budget']:,.2f}
                    </b>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#64748B; margin-top:8px;">
                    <span>Budget Used:</span>
                    <b>{budget_eval['budget_used_pct']:.1f}%</b>
                </div>
            </div>
            <div style="margin-top:18px; background:#E2E8F0; border-radius:8px; height:12px; width:100%; overflow:hidden;">
                <div style="background:{budget_eval['badge_color']}; height:100%; width:{min(100.0, budget_eval['budget_used_pct']):.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with b_col2:
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:24px; border-radius:14px;">
            <h4 style="margin:0 0 10px 0; color:#0F172A;">💡 Financial Advisor Recommendation</h4>
            <p style="color:#334155; font-size:1.0rem; line-height:1.6;">
                {budget_eval['recommendation']}
            </p>
            <div style="margin-top:16px; padding:12px 16px; background:white; border-radius:10px; border:1px solid #E2E8F0;">
                <div style="font-size:0.82rem; font-weight:600; color:#64748B; text-transform:uppercase;">Recommended Daily Limit</div>
                <div style="font-size:1.4rem; font-weight:800; color:#2563EB; margin-top:2px;">
                    ₹{budget_eval['recommended_daily_limit']:,.2f} / day
                </div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">Calculated for safe spending over remaining 15 days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#94A3B8; font-size:0.85rem;'>Smart Expense & Spending Analyzer • Built with Python, Scikit-Learn, Pandas & Streamlit</div>", unsafe_allow_html=True)
