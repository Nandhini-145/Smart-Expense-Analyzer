"""
Future Expense Forecasting Module (Linear Regression)
Module: src/forecasting.py
Author: Antigravity AI
Description: Aggregates historical transaction data into monthly financial sequences,
constructs lag and trend features, and trains a Linear Regression model to forecast upcoming
monthly expenses. Evaluates performance using MAE, MSE, RMSE, and R2 score.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import load_and_preprocess_data

MODEL_DIR = "models"
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.pkl")

def prepare_monthly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups transactional expense logs into monthly time-series aggregations
    and constructs predictive features:
    - month_idx: Continuous sequential integer (1, 2, 3... T).
    - total_amount: Target variable Y (total monthly spending in ₹).
    - tx_count: Total transaction volume in the month.
    - prev_spending: Lag-1 monthly spending (₹).
    - rolling_mean_2m: 2-month rolling average spending.
    """
    # Group by Year-Month
    monthly = df.groupby(["year", "month"]).agg(
        total_amount=("amount", "sum"),
        tx_count=("amount", "count"),
        avg_tx_amt=("amount", "mean")
    ).reset_index()
    
    monthly = monthly.sort_values(by=["year", "month"]).reset_index(drop=True)
    monthly["month_idx"] = np.arange(1, len(monthly) + 1)
    
    # Lag and trend features
    monthly["prev_spending"] = monthly["total_amount"].shift(1).fillna(monthly["total_amount"].iloc[0])
    monthly["rolling_mean_2m"] = monthly["total_amount"].rolling(window=2, min_periods=1).mean()
    
    # Month name string for visualization
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["month_label"] = [f"{month_names[(m-1)%12]} {y}" for m, y in zip(monthly["month"], monthly["year"])]
    
    return monthly

def train_forecast_model(data_path="data/expenses.csv", save_model=True):
    """
    Trains and evaluates the Linear Regression expense forecast model.
    Computes MAE, MSE, RMSE, and R2 score.
    Generates an actual vs forecasted spending plot with next-month projection.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_and_preprocess_data(data_path)
    monthly_df = prepare_monthly_dataset(df)
    
    # Features (X) and Target (y)
    feature_cols = ["month_idx", "prev_spending", "tx_count"]
    X = monthly_df[feature_cols]
    y = monthly_df["total_amount"]
    
    # Train / Test split (Last 3 months as test validation)
    split_idx = max(len(monthly_df) - 3, 2)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model on training set to evaluate generalization
    eval_model = LinearRegression()
    eval_model.fit(X_train, y_train)
    y_test_pred = eval_model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    
    print("=" * 60)
    print("📈 TRAINING FUTURE EXPENSE PREDICTION MODEL (LINEAR REGRESSION)")
    print("=" * 60)
    print(f"Total Historical Months : {len(monthly_df)}")
    print(f"Training Months         : {len(X_train)}")
    print(f"Validation Months       : {len(X_test)}")
    print("\n✅ VALIDATION PERFORMANCE METRICS:")
    print(f"  • Mean Absolute Error (MAE)     : ₹{mae:,.2f}")
    print(f"  • Mean Squared Error (MSE)      : {mse:,.2f}")
    print(f"  • Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
    print(f"  • R² Score (Variance Explained) : {r2:.4f}")
    
    # Train final model on entire historical dataset for forward forecasting
    full_model = LinearRegression()
    full_model.fit(X, y)
    
    # Predict next month (Month N+1)
    last_row = monthly_df.iloc[-1]
    next_month_idx = last_row["month_idx"] + 1
    next_prev_spending = last_row["total_amount"]
    next_tx_count = round(monthly_df["tx_count"].mean())
    
    X_next = pd.DataFrame([{
        "month_idx": next_month_idx,
        "prev_spending": next_prev_spending,
        "tx_count": next_tx_count
    }])
    
    next_month_pred = float(full_model.predict(X_next)[0])
    print(f"\n🔮 NEXT MONTH SPENDING FORECAST:")
    print(f"  ↳ Projected Next Month Expense: ₹{next_month_pred:,.2f}")
    
    # Historical + In-sample predictions
    monthly_df["predicted_amount"] = full_model.predict(X)
    
    # Generate Forecast Visualization Plot
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["month_label"], monthly_df["total_amount"],
             marker="o", label="Actual Monthly Spending", color="#2563EB", linewidth=2.5)
    plt.plot(monthly_df["month_label"], monthly_df["predicted_amount"],
             marker="s", linestyle="--", label="Model Fitted Spending", color="#10B981", linewidth=2)
    
    # Plot future projection
    next_label = f"Month {next_month_idx} (Forecast)"
    plt.plot([monthly_df["month_label"].iloc[-1], next_label],
             [monthly_df["total_amount"].iloc[-1], next_month_pred],
             marker="*", markersize=12, linestyle=":", color="#EF4444", label=f"Next Month Forecast: ₹{next_month_pred:,.0f}")
    
    plt.title("Historical Spending & Future Expense Forecast (Linear Regression)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Timeline", fontsize=10)
    plt.ylabel("Monthly Total Spending (₹)", fontsize=10)
    plt.xticks(rotation=30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "forecast_trend_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Forecast Trend Plot to: {plot_path}")
    
    # Save Model Artifact Bundle (Model + Metadata)
    if save_model:
        bundle = {
            "model": full_model,
            "feature_cols": feature_cols,
            "metrics": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2)
            },
            "last_month_idx": int(last_row["month_idx"]),
            "avg_tx_count": float(next_tx_count)
        }
        joblib.dump(bundle, FORECAST_MODEL_PATH)
        print(f"Saved Forecast Model Bundle to: {FORECAST_MODEL_PATH}")
        
    return full_model, monthly_df, next_month_pred, {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

def load_forecast_model():
    """
    Loads persisted forecast model bundle or trains one if missing.
    """
    if not os.path.exists(FORECAST_MODEL_PATH):
        train_forecast_model()
    return joblib.load(FORECAST_MODEL_PATH)

def forecast_next_months(df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    """
    Forecasts spending for the next N future months given a transactional dataset.
    Returns a DataFrame with projected month, index, and predicted expense.
    """
    bundle = load_forecast_model()
    model = bundle["model"]
    
    monthly_df = prepare_monthly_dataset(df)
    last_row = monthly_df.iloc[-1]
    
    curr_idx = int(last_row["month_idx"])
    curr_spending = float(last_row["total_amount"])
    avg_tx = round(monthly_df["tx_count"].mean())
    
    forecasts = []
    for step in range(1, n_months + 1):
        future_idx = curr_idx + step
        X_fut = pd.DataFrame([{
            "month_idx": future_idx,
            "prev_spending": curr_spending,
            "tx_count": avg_tx
        }])
        pred_amt = max(0.0, float(model.predict(X_fut)[0]))
        forecasts.append({
            "future_month_step": f"+{step} Month",
            "future_index": future_idx,
            "predicted_spending": round(pred_amt, 2)
        })
        curr_spending = pred_amt  # autoregressive rollover
        
    return pd.DataFrame(forecasts)

if __name__ == "__main__":
    model, monthly, next_pred, metrics = train_forecast_model()
    print("\n🔮 Multi-Month Future Projection:")
    fut_df = forecast_next_months(load_and_preprocess_data("data/expenses.csv"), n_months=3)
    print(fut_df)
