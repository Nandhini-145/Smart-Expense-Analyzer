"""
AI Expense Categorization Module
Module: src/categorization.py
Author: Antigravity AI
Description: Trains and evaluates a TF-IDF + Logistic Regression NLP pipeline to classify
expense descriptions into financial categories (Food, Shopping, Transport, etc.).
Saves and loads the trained pipeline to/from models/category_model.pkl.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
# Ensure project root is in sys.path and handle Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, load_and_preprocess_data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "category_model.pkl")

def build_category_pipeline():
    """
    Constructs an end-to-end Scikit-Learn pipeline with:
    1. TfidfVectorizer (unigrams & bigrams, sublinear TF scaling).
    2. LogisticRegression (multi-class with L2 regularization).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=2500,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipeline

def train_category_model(data_path="data/expenses.csv", save_model=True):
    """
    Trains the category prediction model, computes evaluation metrics,
    prints classification report, saves confusion matrix plot, and persists model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load and clean dataset
    df = load_and_preprocess_data(data_path)
    
    # Features (X) and Target (y)
    X = df["clean_description"]
    y = df["category"]
    
    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("=" * 60)
    print("🚀 TRAINING AI EXPENSE CATEGORIZATION MODEL (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)
    print(f"Total Transactions : {len(df)}")
    print(f"Training Samples   : {len(X_train)}")
    print(f"Testing Samples    : {len(X_test)}")
    print(f"Unique Categories  : {y.nunique()} -> {sorted(y.unique().tolist())}")
    
    # 2. Train Pipeline
    pipeline = build_category_pipeline()
    pipeline.fit(X_train, y_train)
    
    # 3. Model Evaluation on Test Set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n✅ MODEL EVALUATION METRICS:")
    print(f"Accuracy Score: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 4. Confusion Matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Save Confusion Matrix visualization
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Expense Categorization Confusion Matrix (Accuracy: {acc*100:.1f}%)")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "category_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix Plot to: {cm_path}")
    
    # 5. Persist Model
    if save_model:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Model successfully saved to: {MODEL_PATH}")
        
    return pipeline, acc

def load_category_model():
    """
    Loads persisted category model pipeline from disk.
    If not found, trains and saves it automatically.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model artifact not found at {MODEL_PATH}. Initiating automatic training...")
        pipeline, _ = train_category_model()
        return pipeline
    return joblib.load(MODEL_PATH)

def predict_category(description: str, model=None):
    """
    Inference function: takes a raw transaction description string and returns
    the predicted category and confidence probability.
    
    Example:
        predict_category("Swiggy chicken biryani")
        -> ("Food", 0.94)
    """
    if model is None:
        model = load_category_model()
        
    cleaned = clean_text(description)
    if not cleaned:
        return "Others", 0.0
        
    pred = model.predict([cleaned])[0]
    proba = np.max(model.predict_proba([cleaned])[0])
    return pred, round(float(proba), 4)

if __name__ == "__main__":
    train_category_model()
    
    # Test sample inference queries
    test_queries = [
        "Swiggy food order dinner",
        "Amazon laptop wireless mouse",
        "Petrol bunk fuel refill",
        "Netflix monthly subscription",
        "Electricity bill Bescom",
        "Apollo Pharmacy medicine",
        "Udemy Python data science bootcamp",
        "H&M summer t-shirt"
    ]
    
    print("\n🔍 SAMPLE INFERENCE PREDICTIONS:")
    print("-" * 55)
    for q in test_queries:
        cat, conf = predict_category(q)
        print(f"Description: '{q}'\n  ↳ Predicted Category: [{cat}] (Confidence: {conf*100:.1f}%)")
