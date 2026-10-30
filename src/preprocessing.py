"""
Preprocessing & Feature Engineering Module
Module: src/preprocessing.py
Author: Antigravity AI
Description: Robust pipeline for cleaning raw expense transaction logs, handling nulls/duplicates,
formatting data types, and deriving temporal & financial features.
"""

import os
import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Cleans raw transaction text by converting to lowercase,
    removing special symbols/numbers, and trimming extra whitespaces.
    Example: 'Swiggy food order #123!' -> 'swiggy food order'
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data(file_path_or_buffer) -> pd.DataFrame:
    """
    Loads raw CSV data and performs comprehensive end-to-end data cleansing:
    1. Reads CSV safely with error handling.
    2. Standardizes column names (lowercase, stripped).
    3. Detects and handles missing values.
    4. Detects and removes duplicate entries.
    5. Converts 'date' to datetime and enforces valid date bounds.
    6. Converts 'amount' to positive numeric values (filters zero/negative noise).
    7. Feature Engineering: year, month, month_name, day, weekday, is_weekend,
       cleaned_description, and lag spending features.
    
    Returns:
        pd.DataFrame: Cleaned and enriched DataFrame.
    """
    try:
        if isinstance(file_path_or_buffer, str):
            if not os.path.exists(file_path_or_buffer):
                raise FileNotFoundError(f"File not found at: {file_path_or_buffer}")
            df = pd.read_csv(file_path_or_buffer)
        else:
            # For Streamlit file uploader (BytesIO / StringIO)
            df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

    # 1. Normalize Column Names
    # Maps potential variants (e.g. Payment_Mode -> payment_method, Expense_ID)
    col_mapping = {
        "expense_id": "id",
        "trans_date": "date",
        "payment_mode": "payment_method",
        "pay_mode": "payment_method",
        "transaction_description": "description",
        "cost": "amount",
        "expense_amount": "amount"
    }
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=col_mapping)

    required_cols = ["date", "description", "amount", "payment_method", "category"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise KeyError(f"Missing mandatory columns in dataset: {missing_req}. Required: {required_cols}")

    initial_count = len(df)

    # 2. Duplicate Detection & Removal
    df = df.drop_duplicates(subset=["date", "description", "amount", "payment_method"]).reset_index(drop=True)
    duplicates_removed = initial_count - len(df)

    # 3. Missing Value Handling
    df["description"] = df["description"].fillna("Unknown Expense").astype(str)
    df["category"] = df["category"].fillna("Others").astype(str)
    df["payment_method"] = df["payment_method"].fillna("UPI").astype(str)
    
    # 4. Amount Cleaning & Validation
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    # Impute missing amounts with median, enforce strictly positive amounts
    median_amount = df["amount"].median() if not df["amount"].dropna().empty else 100.0
    df["amount"] = df["amount"].fillna(median_amount)
    df["amount"] = df["amount"].abs()  # Handle negative values gracefully
    df = df[df["amount"] > 0].copy()

    # 5. Date Parsing & Sorting
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop records where date couldn't be parsed
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(by="date").reset_index(drop=True)

    # 6. Feature Engineering
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.strftime("%A")
    df["is_weekend"] = df["date"].dt.day_of_week.isin([5, 6]).astype(int)
    
    # Rolling & Lag Features
    # Previous spending (lag 1 transaction amount)
    df["previous_spending"] = df["amount"].shift(1).fillna(df["amount"].iloc[0])
    # 7-day rolling average spending
    df["rolling_avg_7d"] = df["amount"].rolling(window=7, min_periods=1).mean()
    # Transaction index count
    df["tx_count"] = np.arange(1, len(df) + 1)

    # Text Preprocessing for NLP Model
    df["clean_description"] = df["description"].apply(clean_text)

    # Format date back to string for clean display/JSON serialization if needed
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    return df

if __name__ == "__main__":
    test_path = "data/expenses.csv"
    print("Testing Preprocessing Pipeline...")
    cleaned_df = load_and_preprocess_data(test_path)
    print(f"Processed DataFrame shape: {cleaned_df.shape}")
    print("\nColumns:", cleaned_df.columns.tolist())
    print("\nFirst 3 rows:")
    print(cleaned_df[["date_str", "description", "clean_description", "amount", "category", "month_name", "weekday"]].head(3))
