"""
PostgreSQL Database Integration Module
Module: src/database.py
Author: Antigravity AI
Description: Manages PostgreSQL relational database connection, table creation, parameterized
CRUD operations, bulk ingestion of cleaned expense logs, and analytical aggregate queries.
Includes automatic SQLite fallback if PostgreSQL server credentials are not configured locally.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psycopg2
from psycopg2 import sql, extras
import pandas as pd
import sqlite3

class ExpenseDatabaseManager:
    """
    Production Database Manager supporting PostgreSQL with parameterized queries,
    bulk ingestion, CRUD operations, and analytical aggregation.
    """
    def __init__(self, dbname="expense_db", user="postgres", password="password", host="localhost", port=5432):
        self.dbname = os.getenv("DB_NAME", dbname)
        self.user = os.getenv("DB_USER", user)
        self.password = os.getenv("DB_PASSWORD", password)
        self.host = os.getenv("DB_HOST", host)
        self.port = os.getenv("DB_PORT", port)
        
        self.is_postgres = False
        self.conn = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Attempts PostgreSQL connection; falls back gracefully to local SQLite if PG is offline."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                connect_timeout=3
            )
            self.conn.autocommit = True
            self.is_postgres = True
            print(f"✅ Successfully connected to PostgreSQL Database [{self.dbname}] on {self.host}:{self.port}")
        except Exception as e:
            # Fallback to local SQLite for seamless offline testing
            print(f"⚠️ PostgreSQL not reachable ({e}). Initializing embedded SQLite database 'expense_db.sqlite' for zero-friction execution.")
            self.conn = sqlite3.connect("data/expense_db.sqlite", check_same_thread=False)
            self.is_postgres = False

    def create_table(self):
        """Creates the 'expenses' table with proper primary keys, types, and constraints."""
        cursor = self.conn.cursor()
        if self.is_postgres:
            create_query = """
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                description VARCHAR(255) NOT NULL,
                amount NUMERIC(10, 2) NOT NULL CHECK (amount > 0),
                payment_method VARCHAR(50) NOT NULL,
                category VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);
            CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category);
            """
        else:
            create_query = """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL CHECK (amount > 0),
                payment_method TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        cursor.execute(create_query)
        if not self.is_postgres:
            self.conn.commit()
        cursor.close()
        print("✅ 'expenses' table created/verified successfully.")

    def insert_transaction(self, date_str: str, description: str, amount: float, payment_method: str, category: str):
        """Parameterized insert of a single expense record."""
        cursor = self.conn.cursor()
        placeholder = "%s" if self.is_postgres else "?"
        query = f"""
            INSERT INTO expenses (date, description, amount, payment_method, category)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """
        params = (date_str, description, float(amount), payment_method, category)
        cursor.execute(query, params)
        if not self.is_postgres:
            self.conn.commit()
        cursor.close()

    def bulk_insert_dataframe(self, df: pd.DataFrame):
        """Efficient batch insertion of an entire cleaned DataFrame."""
        self.create_table()
        cursor = self.conn.cursor()
        records = [
            (row["date_str"] if "date_str" in row else str(row["date"]),
             row["description"], float(row["amount"]), row["payment_method"], row["category"])
            for _, row in df.iterrows()
        ]
        
        if self.is_postgres:
            insert_query = """
                INSERT INTO expenses (date, description, amount, payment_method, category)
                VALUES %s
            """
            extras.execute_values(cursor, insert_query, records)
        else:
            cursor.executemany(
                "INSERT INTO expenses (date, description, amount, payment_method, category) VALUES (?, ?, ?, ?, ?)",
                records
            )
            self.conn.commit()
            
        cursor.close()
        print(f"✅ Bulk inserted {len(records)} transactions into 'expenses' table.")

    def select_all_expenses(self, limit: int = 1000) -> pd.DataFrame:
        """Selects all transactions up to limit."""
        query = f"SELECT id, date, description, amount, payment_method, category FROM expenses ORDER BY date DESC LIMIT {limit}"
        return pd.read_sql_query(query, self.conn)

    def select_by_category(self, category: str) -> pd.DataFrame:
        """Parameterized selection of transactions by category."""
        placeholder = "%s" if self.is_postgres else "?"
        query = f"SELECT * FROM expenses WHERE category = {placeholder} ORDER BY date DESC"
        return pd.read_sql_query(query, self.conn, params=[category])

    def update_transaction(self, expense_id: int, new_category: str = None, new_amount: float = None):
        """Parameterized update for a transaction's category or amount."""
        cursor = self.conn.cursor()
        placeholder = "%s" if self.is_postgres else "?"
        
        if new_category and new_amount:
            query = f"UPDATE expenses SET category = {placeholder}, amount = {placeholder} WHERE id = {placeholder}"
            params = (new_category, float(new_amount), expense_id)
        elif new_category:
            query = f"UPDATE expenses SET category = {placeholder} WHERE id = {placeholder}"
            params = (new_category, expense_id)
        elif new_amount:
            query = f"UPDATE expenses SET amount = {placeholder} WHERE id = {placeholder}"
            params = (float(new_amount), expense_id)
        else:
            return
            
        cursor.execute(query, params)
        if not self.is_postgres:
            self.conn.commit()
        cursor.close()
        print(f"✅ Updated expense ID {expense_id}.")

    def delete_transaction(self, expense_id: int):
        """Parameterized deletion of a transaction by ID."""
        cursor = self.conn.cursor()
        placeholder = "%s" if self.is_postgres else "?"
        query = f"DELETE FROM expenses WHERE id = {placeholder}"
        cursor.execute(query, (expense_id,))
        if not self.is_postgres:
            self.conn.commit()
        cursor.close()
        print(f"✅ Deleted expense ID {expense_id}.")

    def get_category_aggregates(self) -> pd.DataFrame:
        """SQL aggregate query: category-wise total spending, transaction counts, and average amount."""
        query = """
            SELECT 
                category,
                COUNT(*) AS transaction_count,
                ROUND(CAST(SUM(amount) AS NUMERIC), 2) AS total_spending,
                ROUND(CAST(AVG(amount) AS NUMERIC), 2) AS average_amount,
                ROUND(CAST(MIN(amount) AS NUMERIC), 2) AS min_amount,
                ROUND(CAST(MAX(amount) AS NUMERIC), 2) AS max_amount
            FROM expenses
            GROUP BY category
            ORDER BY total_spending DESC;
        """
        if not self.is_postgres:
            query = query.replace("CAST(SUM(amount) AS NUMERIC)", "SUM(amount)") \
                         .replace("CAST(AVG(amount) AS NUMERIC)", "AVG(amount)") \
                         .replace("CAST(MIN(amount) AS NUMERIC)", "MIN(amount)") \
                         .replace("CAST(MAX(amount) AS NUMERIC)", "MAX(amount)")
        return pd.read_sql_query(query, self.conn)

    def get_monthly_aggregates(self) -> pd.DataFrame:
        """SQL aggregate query: monthly spending and counts."""
        if self.is_postgres:
            query = """
                SELECT 
                    TO_CHAR(date, 'YYYY-MM') AS month_year,
                    COUNT(*) AS transaction_count,
                    ROUND(CAST(SUM(amount) AS NUMERIC), 2) AS total_monthly_spending
                FROM expenses
                GROUP BY TO_CHAR(date, 'YYYY-MM')
                ORDER BY month_year;
            """
        else:
            query = """
                SELECT 
                    SUBSTR(date, 1, 7) AS month_year,
                    COUNT(*) AS transaction_count,
                    ROUND(SUM(amount), 2) AS total_monthly_spending
                FROM expenses
                GROUP BY SUBSTR(date, 1, 7)
                ORDER BY month_year;
            """
        return pd.read_sql_query(query, self.conn)

    def close(self):
        """Closes database connection safely."""
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    from src.preprocessing import load_and_preprocess_data
    print("Testing Database Module...")
    db = ExpenseDatabaseManager()
    db.create_table()
    
    # Ingest data
    df = load_and_preprocess_data("data/expenses.csv")
    db.bulk_insert_dataframe(df.head(100))
    
    # Test SELECT
    records = db.select_all_expenses(limit=5)
    print("\nRecent 5 Records from Database:")
    print(records)
    
    # Test Aggregates
    print("\nCategory-wise Aggregates from SQL:")
    cat_aggs = db.get_category_aggregates()
    print(cat_aggs)
    
    # Test Insert & Delete
    db.insert_transaction("2025-12-31", "Test New Year Dinner", 2500.0, "UPI", "Food")
    print("Inserted test single transaction.")
    
    db.close()
