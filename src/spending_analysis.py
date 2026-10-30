"""
Spending Analysis & Visualization Module
Module: src/spending_analysis.py
Author: Antigravity AI
Description: Performs deep statistical analysis and visualization of user expenses using Pandas,
NumPy, and Matplotlib. Analyzes totals, averages, category breakdowns, monthly/daily trends,
payment modes, and top expensive transactions.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_and_preprocess_data

def compute_spending_summary(df: pd.DataFrame) -> dict:
    """
    Computes high-level spending summary KPIs:
    Total, average, median, min, max expenses, highest/lowest category, and transaction counts.
    """
    total_exp = float(df["amount"].sum())
    avg_exp = float(df["amount"].mean())
    median_exp = float(df["amount"].median())
    min_exp = float(df["amount"].min())
    max_exp = float(df["amount"].max())
    total_tx = int(len(df))
    
    # Category aggregation
    cat_summary = df.groupby("category")["amount"].agg(["sum", "count"]).sort_values(by="sum", ascending=False)
    highest_cat = cat_summary.index[0]
    highest_cat_amt = float(cat_summary.iloc[0]["sum"])
    lowest_cat = cat_summary.index[-1]
    lowest_cat_amt = float(cat_summary.iloc[-1]["sum"])
    
    # Payment method breakdown
    pm_summary = df.groupby("payment_method")["amount"].agg(["sum", "count"]).sort_values(by="sum", ascending=False)
    
    # Monthly spending
    monthly = df.groupby(["year", "month", "month_name"])["amount"].sum().reset_index()
    monthly = monthly.sort_values(by=["year", "month"])
    
    return {
        "total_expense": round(total_exp, 2),
        "average_expense": round(avg_exp, 2),
        "median_expense": round(median_exp, 2),
        "min_expense": round(min_exp, 2),
        "max_expense": round(max_exp, 2),
        "total_transactions": total_tx,
        "highest_spending_category": highest_cat,
        "highest_spending_amount": round(highest_cat_amt, 2),
        "lowest_spending_category": lowest_cat,
        "lowest_spending_amount": round(lowest_cat_amt, 2),
        "category_summary": cat_summary,
        "payment_method_summary": pm_summary,
        "monthly_summary": monthly,
        "top_10_transactions": df.nlargest(10, "amount")[["date_str", "description", "category", "amount", "payment_method"]]
    }

def plot_spending_analysis(df: pd.DataFrame, output_dir="models/charts"):
    """
    Generates and saves modern, readable spending visualization charts:
    1. Monthly spending trend (Line chart with markers).
    2. Category distribution (Donut chart).
    3. Payment method analysis (Horizontal Bar chart).
    4. Day of the week spending patterns (Bar chart).
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Monthly Spending Trend
    monthly = df.groupby(["month", "month_name"])["amount"].sum().reset_index().sort_values(by="month")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly["month_name"], monthly["amount"], marker="o", linewidth=2.5, color="#2563EB", markersize=6)
    ax.fill_between(monthly["month_name"], monthly["amount"], alpha=0.15, color="#2563EB")
    ax.set_title("Monthly Spending Trend (2025)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Total Spending (₹)", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "monthly_spending_trend.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close()
    
    # 2. Category Distribution (Donut Chart)
    cat_tot = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = sns.color_palette("Set2", len(cat_tot))
    wedges, texts, autotexts = ax.pie(
        cat_tot, labels=cat_tot.index, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78, colors=colors,
        textprops=dict(color="#1E293B", fontsize=10)
    )
    # Donut center circle
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title("Category-wise Spending Share", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "category_spending_donut.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close()
    
    # 3. Payment Method Bar Chart
    pm_tot = df.groupby("payment_method")["amount"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(pm_tot.index, pm_tot.values, color="#10B981", height=0.55)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (pm_tot.max()*0.01), bar.get_y() + bar.get_height()/2,
                f"₹{width:,.0f}", va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_title("Total Spending by Payment Method", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Spending Amount (₹)", fontsize=11)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "payment_method_spending.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close()
    
    # 4. Weekday Spending
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_tot = df.groupby("weekday")["amount"].mean().reindex(weekday_order)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(weekday_tot.index, weekday_tot.values, color="#F59E0B", width=0.55)
    ax.set_title("Average Transaction Amount by Day of Week", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Expense (₹)", fontsize=11)
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "weekday_spending_pattern.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close()

    print(f"All spending visualization charts saved to: {output_dir}")
    return [fig1_path, fig2_path, fig3_path, fig4_path]

if __name__ == "__main__":
    df = load_and_preprocess_data("data/expenses.csv")
    summary = compute_spending_summary(df)
    print("=" * 60)
    print("📊 SPENDING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Expenses        : ₹{summary['total_expense']:,.2f}")
    print(f"Average Expense       : ₹{summary['average_expense']:,.2f}")
    print(f"Median Expense        : ₹{summary['median_expense']:,.2f}")
    print(f"Min Expense           : ₹{summary['min_expense']:,.2f}")
    print(f"Max Expense           : ₹{summary['max_expense']:,.2f}")
    print(f"Total Transactions    : {summary['total_transactions']}")
    print(f"Highest Spending Cat  : {summary['highest_spending_category']} (₹{summary['highest_spending_amount']:,.2f})")
    print(f"Lowest Spending Cat   : {summary['lowest_spending_category']} (₹{summary['lowest_spending_amount']:,.2f})")
    print("\nTop 5 Most Expensive Transactions:")
    print(summary["top_10_transactions"].head(5))
    plot_spending_analysis(df)
