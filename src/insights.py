"""
AI Spending Insights & Smart Budget Recommendation Engine
Module: src/insights.py
Author: Antigravity AI
Description: Generates automated, actionable financial intelligence and budget tracking metrics
from user expense logs without requiring external paid LLM APIs.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data
from src.anomaly import detect_anomalies
from src.forecasting import forecast_next_months

def generate_spending_insights(df: pd.DataFrame) -> list:
    """
    Generates dynamic, rule-based financial insights:
    1. Month-over-month category growth rate.
    2. Primary & secondary spending categories.
    3. Anomaly alert for suspicious or high spikes.
    4. Future expense forecast prediction.
    5. Actionable cost-cutting recommendation (e.g. 10% reduction savings).
    """
    insights = []
    
    if df.empty:
        return ["No transactions available to generate insights."]

    # 1. Highest and Second-Highest Category Analysis
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    if len(cat_totals) > 0:
        top_cat = cat_totals.index[0]
        top_amt = cat_totals.iloc[0]
        top_pct = (top_amt / df["amount"].sum()) * 100
        insights.append(f"🏆 **Primary Spending Driver**: **{top_cat}** represents your highest expenditure at **₹{top_amt:,.2f}** ({top_pct:.1f}% of total spending).")
        
    if len(cat_totals) > 1:
        second_cat = cat_totals.index[1]
        second_amt = cat_totals.iloc[1]
        second_pct = (second_amt / df["amount"].sum()) * 100
        insights.append(f"📊 **Secondary Expenditure**: **{second_cat}** is your second-highest spending category, accounting for **₹{second_amt:,.2f}** ({second_pct:.1f}%).")

    # 2. Month-over-Month Category Trend Comparison
    monthly_cat = df.groupby(["year", "month", "category"])["amount"].sum().unstack(fill_value=0)
    if len(monthly_cat) >= 2:
        last_month = monthly_cat.iloc[-1]
        prev_month = monthly_cat.iloc[-2]
        
        # Calculate percentage changes for categories with meaningful base spending
        diffs = {}
        for c in monthly_cat.columns:
            p_val = prev_month.get(c, 0)
            l_val = last_month.get(c, 0)
            if p_val > 500:
                pct_chg = ((l_val - p_val) / p_val) * 100
                diffs[c] = pct_chg
                
        if diffs:
            max_increase_cat = max(diffs, key=diffs.get)
            inc_val = diffs[max_increase_cat]
            if inc_val > 5:
                insights.append(f"📈 **Spending Surge Alert**: Your **{max_increase_cat}** expenses increased by **{inc_val:.1f}%** compared with the previous month.")
            elif inc_val < -5:
                insights.append(f"📉 **Expense Reduction**: Great job! Your **{max_increase_cat}** expenses decreased by **{abs(inc_val):.1f}%** from last month.")

    # 3. Anomaly Detection Alerts
    df_with_anomalies = detect_anomalies(df)
    anomalies = df_with_anomalies[df_with_anomalies["is_anomaly"] == 1]
    if not anomalies.empty:
        highest_anomaly = anomalies.sort_values(by="amount", ascending=False).iloc[0]
        insights.append(
            f"🚨 **Unusual Transaction Flagged**: An atypical **₹{highest_anomaly['amount']:,.2f}** transaction was detected on **{highest_anomaly['date_str']}** ('{highest_anomaly['description']}'). Total anomalies detected: **{len(anomalies)}**."
        )
    else:
        insights.append("✅ **Clean Spending Log**: No erratic or statistically abnormal spending spikes detected.")

    # 4. Next-Month Expense Forecast
    try:
        fut = forecast_next_months(df, n_months=1)
        next_proj = fut.iloc[0]["predicted_spending"]
        insights.append(f"🔮 **AI Forecast**: Your predicted total spending for next month is approximately **₹{next_proj:,.2f}** based on your historical trajectory.")
    except Exception:
        pass

    # 5. Smart Actionable Cost-Saving Recommendation
    if len(cat_totals) > 0:
        top_cat_for_saving = "Shopping" if "Shopping" in cat_totals.index else cat_totals.index[0]
        cat_spending = cat_totals.get(top_cat_for_saving, cat_totals.iloc[0])
        monthly_avg_cat = cat_spending / max(df["month"].nunique(), 1)
        potential_savings = monthly_avg_cat * 0.10
        insights.append(
            f"💡 **Cost Optimization Tip**: Reducing **{top_cat_for_saving}** expenses by just **10%** may save approximately **₹{potential_savings:,.2f}** per month."
        )

    return insights

def evaluate_smart_budget(monthly_spending: float, monthly_budget: float) -> dict:
    """
    Evaluates monthly spending against user-defined budget.
    Categories:
    - Below 70%   -> Safe 🟢
    - 70% to 90%  -> Moderate 🟡
    - 90% to 100% -> Warning 🟠
    - Above 100%  -> Over Budget 🔴 (⚠️ Over Budget)
    """
    if monthly_budget <= 0:
        monthly_budget = 25000.0  # sensible default

    budget_used_pct = (monthly_spending / monthly_budget) * 100
    remaining_budget = monthly_budget - monthly_spending
    overspending_amt = max(0.0, monthly_spending - monthly_budget)
    
    if budget_used_pct < 70.0:
        status = "Safe"
        badge_color = "#10B981"  # Emerald Green
        icon = "🟢"
        recommendation = "You are spending comfortably within your limits. Great financial discipline!"
    elif 70.0 <= budget_used_pct < 90.0:
        status = "Moderate"
        badge_color = "#F59E0B"  # Amber
        icon = "🟡"
        recommendation = "Your spending is on track, but monitor discretionary purchases closely."
    elif 90.0 <= budget_used_pct <= 100.0:
        status = "Warning"
        badge_color = "#F97316"  # Orange
        icon = "🟠"
        recommendation = "Approaching budget limit! Curb non-essential spending for the rest of the month."
    else:
        status = "Over Budget"
        badge_color = "#EF4444"  # Red
        icon = "⚠️"
        recommendation = f"You have exceeded your monthly budget by ₹{overspending_amt:,.2f}! Immediate cost cutback recommended."

    # Recommended daily spending limit for remaining days (assume 30-day month)
    daily_limit = max(0.0, remaining_budget / 15.0) if remaining_budget > 0 else 0.0

    return {
        "budget": round(monthly_budget, 2),
        "actual_spending": round(monthly_spending, 2),
        "remaining_budget": round(remaining_budget, 2),
        "overspending_amount": round(overspending_amt, 2),
        "budget_used_pct": round(budget_used_pct, 2),
        "status": status,
        "badge_color": badge_color,
        "icon": icon,
        "recommendation": recommendation,
        "recommended_daily_limit": round(daily_limit, 2)
    }

if __name__ == "__main__":
    df = load_and_preprocess_data("data/expenses.csv")
    print("=" * 60)
    print("🧠 AI-GENERATED SPENDING INSIGHTS")
    print("=" * 60)
    insights = generate_spending_insights(df)
    for idx, ins in enumerate(insights, 1):
        print(f"{idx}. {ins}")
        
    print("\n" + "=" * 60)
    print("🎯 SMART BUDGET EVALUATION TEST")
    print("=" * 60)
    test_cases = [
        (15000, 25000),   # 60% -> Safe
        (20000, 25000),   # 80% -> Moderate
        (23500, 25000),   # 94% -> Warning
        (27500, 25000),   # 110% -> Over Budget
    ]
    for actual, budget in test_cases:
        res = evaluate_smart_budget(actual, budget)
        print(f"Budget: ₹{res['budget']:,.0f} | Actual: ₹{res['actual_spending']:,.0f} | Used: {res['budget_used_pct']:>5.1f}% | Status: {res['icon']} {res['status']:<12} | Rem: ₹{res['remaining_budget']:>9,.2f}")
