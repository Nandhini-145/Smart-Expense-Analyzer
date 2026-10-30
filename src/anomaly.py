"""
AI Anomaly Detection Module (Isolation Forest)
Module: src/anomaly.py
Author: Antigravity AI
Description: Trains an unsupervised Isolation Forest model to detect unusual, suspicious, or
statistically rare spending spikes. Persists the trained model to models/anomaly_model.pkl.
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
import seaborn as sns
from sklearn.ensemble import IsolationForest

from src.preprocessing import load_and_preprocess_data

MODEL_DIR = "models"
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")

def train_anomaly_model(data_path="data/expenses.csv", contamination=0.025, save_model=True):
    """
    Trains an Isolation Forest model on transaction amounts to detect sudden high-value outliers.
    
    Why Isolation Forest?
    - Unsupervised algorithm: No need for manual anomaly labels.
    - Isolates outliers by recursively building random decision trees.
    - Outliers have extreme values and require fewer splits to isolate (shorter path length).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_and_preprocess_data(data_path)
    
    # Feature for Isolation Forest (2D array of amounts)
    X = df[["amount"]].values
    
    # Isolation Forest setup
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X)
    
    # Predictions: 1 = Normal, -1 = Anomaly
    preds = iso_forest.predict(X)
    scores = iso_forest.decision_function(X)
    
    df["is_anomaly"] = np.where(preds == -1, 1, 0)
    df["anomaly_score"] = scores
    
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_pct = (anomaly_count / len(df)) * 100
    
    print("=" * 60)
    print("🚨 TRAINING AI ANOMALY DETECTION MODEL (ISOLATION FOREST)")
    print("=" * 60)
    print(f"Total Transactions Analyzed : {len(df)}")
    print(f"Normal Transactions         : {len(df) - anomaly_count}")
    print(f"Anomalous Transactions      : {anomaly_count}")
    print(f"Anomaly Rate                : {anomaly_pct:.2f}%")
    
    # Display flagged anomalies
    anomalies = df[df["is_anomaly"] == 1][["date_str", "description", "category", "amount", "payment_method"]]
    print("\n🔍 Flagged Anomalous Spikes:")
    print(anomalies.head(10))
    
    # Plot Scatter Anomaly Visualization
    plt.figure(figsize=(10, 5))
    plt.scatter(
        df[df["is_anomaly"] == 0].index,
        df[df["is_anomaly"] == 0]["amount"],
        c="#3B82F6", label="Normal Transactions", alpha=0.6, s=25
    )
    plt.scatter(
        df[df["is_anomaly"] == 1].index,
        df[df["is_anomaly"] == 1]["amount"],
        c="#EF4444", label="Detected Anomalies", alpha=0.9, s=70, marker="X"
    )
    plt.title(f"Expense Anomaly Detection via Isolation Forest ({anomaly_count} Anomalies Flagged)", fontsize=12, fontweight="bold")
    plt.xlabel("Transaction Index", fontsize=10)
    plt.ylabel("Expense Amount (₹)", fontsize=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "anomaly_detection_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved Anomaly Scatter Plot to: {plot_path}")
    
    if save_model:
        joblib.dump(iso_forest, ANOMALY_MODEL_PATH)
        print(f"Saved Anomaly Model to: {ANOMALY_MODEL_PATH}")
        
    return iso_forest, df

def load_anomaly_model():
    """
    Loads saved Isolation Forest model or trains one if missing.
    """
    if not os.path.exists(ANOMALY_MODEL_PATH):
        model, _ = train_anomaly_model()
        return model
    return joblib.load(ANOMALY_MODEL_PATH)

def detect_anomalies(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Applies the Isolation Forest model to an expense DataFrame and attaches:
    - 'is_anomaly': 1 if anomalous spike, 0 if normal.
    - 'status': 'Anomaly ⚠️' or 'Normal ✅'
    - 'anomaly_score': Decision function score.
    """
    if model is None:
        model = load_anomaly_model()
        
    X = df[["amount"]].values
    preds = model.predict(X)
    scores = model.decision_function(X)
    
    df = df.copy()
    df["is_anomaly"] = np.where(preds == -1, 1, 0)
    df["status"] = np.where(preds == -1, "Anomaly ⚠️", "Normal ✅")
    df["anomaly_score"] = scores
    return df

def check_single_transaction_anomaly(amount: float, model=None) -> dict:
    """
    Evaluates whether a single transaction amount is flagged as an unusual outlier.
    """
    if model is None:
        model = load_anomaly_model()
        
    pred = model.predict([[amount]])[0]
    score = float(model.decision_function([[amount]])[0])
    is_ano = bool(pred == -1)
    
    return {
        "amount": amount,
        "is_anomaly": is_ano,
        "status": "Anomaly ⚠️" if is_ano else "Normal ✅",
        "anomaly_score": round(score, 4),
        "explanation": "Amount significantly exceeds typical spending patterns" if is_ano else "Amount aligns with normal spending behavior"
    }

if __name__ == "__main__":
    train_anomaly_model()
    
    # Test single checks
    print("\n🔬 TESTING SINGLE AMOUNTS:")
    test_amounts = [250.0, 450.0, 1500.0, 8500.0, 25000.0, 68000.0]
    for amt in test_amounts:
        res = check_single_transaction_anomaly(amt)
        print(f"Amount ₹{res['amount']:>8,.2f} -> Status: {res['status']:<12} | Score: {res['anomaly_score']:>7.4f} ({res['explanation']})")
