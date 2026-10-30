"""
Dataset Generator for Smart Expense & Spending Analyzer
Generates 1,000 realistic expense transactions with descriptions, categories,
realistic amounts, dates across 12 months (2025), payment methods, and realistic anomalies.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_expenses_dataset(output_path="data/expenses.csv", num_records=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Categories and realistic merchant/item descriptions
    catalog = {
        "Food": {
            "descriptions": [
                "Swiggy food order", "Zomato dinner delivery", "Starbucks coffee & snack",
                "Domino's pizza party", "Grocery shopping bigbasket", "Blinkit quick snacks",
                "McDonald's burger meal", "Cafe Coffee Day latte", "Local fruit and vegetable vendor",
                "Subway sandwich lunch", "Barbeque Nation buffet", "KFC chicken bucket",
                "Haldiram sweet snacks", "Bakery fresh bread cakes", "Organic kitchen milk curd"
            ],
            "min_amt": 80, "max_amt": 1500, "anomaly_prob": 0.015
        },
        "Shopping": {
            "descriptions": [
                "Zara casual clothing", "H&M summer t-shirt", "Amazon fashion wardrobe",
                "Myntra festival shoes", "Ajio ethnic wear kurti", "Nike running shoes",
                "Decathlon sports gear", "Levis denim jeans", "Shoppers Stop retail shopping",
                "Titan wrist watch", "Wildcraft backpack travel", "Westside formal shirt trousers"
            ],
            "min_amt": 600, "max_amt": 5500, "anomaly_prob": 0.03
        },
        "Transport": {
            "descriptions": [
                "Petrol bunk fuel refill", "Uber city cab ride", "Ola auto commute",
                "Metro smart card recharge", "Indian Railways train ticket", "Shell petrol bunk diesel",
                "Flight ticket MakeMyTrip", "Airport parking charges", "Fastag toll plaza payment",
                "Two-wheeler vehicle servicing", "Rapido bike taxi ride", "Bus transit daily pass"
            ],
            "min_amt": 40, "max_amt": 1800, "anomaly_prob": 0.02
        },
        "Entertainment": {
            "descriptions": [
                "Netflix monthly subscription", "BookMyShow cinema tickets", "Spotify premium annual",
                "Amazon Prime video renew", "PlayStation store games", "Hotstar cricket subscription",
                "Concert music show pass", "Bowling alley amusement park", "YouTube Premium family plan",
                "Gaming arcade zone card", "Standup comedy open mic ticket", "Museum and gallery entry"
            ],
            "min_amt": 149, "max_amt": 2200, "anomaly_prob": 0.01
        },
        "Utilities": {
            "descriptions": [
                "Electricity bill Bescom", "Airtel broadband wifi bill", "PNG cooking gas cylinder",
                "Water supply municipal bill", "Jio mobile postpaid bill", "Apartment maintenance society",
                "Tata Play DTH recharge", "House cleaning deep sanitization", "Garbage collection civic fee",
                "Solar geyser maintenance"
            ],
            "min_amt": 350, "max_amt": 4500, "anomaly_prob": 0.02
        },
        "Healthcare": {
            "descriptions": [
                "Apollo Pharmacy medicines", "Practo doctor consultation", "Dr Lal PathLabs blood test",
                "Dental clinic scaling treatment", "MedPlus health wellness vitamins", "Eye clinic glasses frames",
                "Physiotherapy rehabilitation session", "Skin specialist derma cream", "1mg health diagnostic check",
                "First aid essentials kit"
            ],
            "min_amt": 150, "max_amt": 3500, "anomaly_prob": 0.02
        },
        "Electronics": {
            "descriptions": [
                "Croma portable powerbank", "Amazon wireless mouse keyboard", "Noise smartwatch fitness tracker",
                "Boat bluetooth headphones", "SanDisk external SSD drive", "Apple lightning charging cable",
                "Mi home smart security camera", "Monitor screen HDMI converter", "Multi-plug spike surge protector"
            ],
            "min_amt": 450, "max_amt": 4900, "anomaly_prob": 0.04
        },
        "Education": {
            "descriptions": [
                "Udemy Python data science course", "Coursera ML specialization fee", "Kindle technical ebook",
                "College semester examination fee", "Cambridge English exam prep book", "Notion AI productivity tool",
                "DataCamp yearly learning subscription", "Stationery notebooks pens spiral", "Research paper IEEE access"
            ],
            "min_amt": 299, "max_amt": 4200, "anomaly_prob": 0.02
        },
        "Others": {
            "descriptions": [
                "Temple charity donation trust", "Birthday gift friend packaging", "Home plumbing faucet repair",
                "Dry cleaning laundry service", "Pet care dog food vaccine", "Car wash doorstep vacuum",
                "Courier postal speed service", "Office team contribution gift"
            ],
            "min_amt": 100, "max_amt": 2500, "anomaly_prob": 0.01
        }
    }
    
    payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]
    payment_weights = [0.45, 0.25, 0.15, 0.08, 0.07]
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days
    
    records = []
    categories = list(catalog.keys())
    # Frequency weights across categories
    cat_weights = [0.24, 0.16, 0.14, 0.10, 0.11, 0.08, 0.07, 0.06, 0.04]
    
    for i in range(1, num_records + 1):
        # Pick category
        cat = random.choices(categories, weights=cat_weights, k=1)[0]
        cat_info = catalog[cat]
        
        # Pick description
        desc = random.choice(cat_info["descriptions"])
        
        # Determine amount (normal vs anomaly outlier)
        is_anomaly = random.random() < cat_info["anomaly_prob"]
        if is_anomaly:
            # Sudden high spike: ₹12,000 to ₹78,000
            amt = round(random.uniform(12000, 78000), 2)
            # Add context to description sometimes
            if cat == "Electronics":
                desc = "Apple MacBook Air M3 laptop"
            elif cat == "Healthcare":
                desc = "Emergency hospitalization & surgery bill"
            elif cat == "Shopping":
                desc = "Tanishq 22K gold jewelry shopping"
            elif cat == "Transport":
                desc = "MakeMyTrip international round-trip flights"
        else:
            # Base amount with slight log-normal variation
            base = random.uniform(cat_info["min_amt"], cat_info["max_amt"])
            # Slight upward trend towards late year to demonstrate realistic inflation / forecasting pattern
            month_factor = 1.0 + (random.randint(0, date_range_days) / date_range_days) * 0.15
            amt = round(base * month_factor, 2)
            
        # Pick date
        random_day = random.randint(0, date_range_days)
        tx_date = (start_date + timedelta(days=random_day)).strftime("%Y-%m-%d")
        
        # Pick payment mode
        pm = random.choices(payment_methods, weights=payment_weights, k=1)[0]
        
        records.append({
            "date": tx_date,
            "description": desc,
            "amount": amt,
            "payment_method": pm,
            "category": cat
        })
        
    df = pd.DataFrame(records)
    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} records at: {output_path}")
    print("\nDataset Overview:")
    print(df.info())
    print("\nCategory Distribution:")
    print(df["category"].value_counts())
    print("\nSample Rows:")
    print(df.head(5))
    return df

if __name__ == "__main__":
    generate_expenses_dataset("data/expenses.csv", num_records=1000)
