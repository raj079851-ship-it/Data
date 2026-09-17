"""
Demo SQLite Database Generator
Creates lightweight, realistic SQLite relational databases for instant live demonstration
of database connections, multi-table joins, and schema discovery.
"""

import os
import sqlite3
from typing import Tuple


def get_or_create_demo_sqlite() -> Tuple[str, str]:
    """
    Creates and seeds a demo SQLite database in the workspace with multiple relational tables.
    Returns (db_path, description).
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sample_data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "enterprise_demo.db")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Orders Table
    cur.execute("DROP TABLE IF EXISTS orders;")
    cur.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_category TEXT NOT NULL,
            order_amount REAL NOT NULL,
            payment_status TEXT NOT NULL,
            order_date DATE NOT NULL
        );
    """)

    # 2. Customers Table
    cur.execute("DROP TABLE IF EXISTS customers;")
    cur.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL,
            country TEXT NOT NULL,
            tier TEXT NOT NULL,
            signup_year INTEGER NOT NULL
        );
    """)

    # Seed data
    customers_data = [
        (101, "Nexus Retail", "USA", "Enterprise", 2023),
        (102, "Solaris GmbH", "Germany", "Mid-Market", 2022),
        (103, "Tokyo Horizon", "Japan", "Enterprise", 2021),
        (104, "Aura Systems", "UK", "Growth", 2024),
        (105, "Nordic Analytics", "Sweden", "Mid-Market", 2023),
        (106, "Australis Ltd", "Australia", "Enterprise", 2020),
        (107, "Maple Logistics", "Canada", "Growth", 2024),
        (108, "Sahara Tech", "UAE", "Mid-Market", 2023),
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?);", customers_data)

    orders_data = [
        (1001, 101, "Enterprise AI Suite", 45000.0, "Paid", "2026-01-15"),
        (1002, 101, "Cloud Compute Pack", 12500.0, "Paid", "2026-02-10"),
        (1003, 102, "Data Warehouse Sync", 18200.0, "Paid", "2026-01-20"),
        (1004, 103, "Enterprise AI Suite", 92000.0, "Paid", "2026-02-05"),
        (1005, 104, "Growth Pipeline", 6500.0, "Pending", "2026-02-18"),
        (1006, 105, "Cloud Compute Pack", 14300.0, "Paid", "2026-01-28"),
        (1007, 106, "Enterprise AI Suite", 110000.0, "Paid", "2026-02-12"),
        (1008, 107, "Growth Pipeline", 7200.0, "Paid", "2026-02-22"),
        (1009, 108, "Data Warehouse Sync", 21500.0, "Refunded", "2026-01-10"),
        (1010, 103, "Cloud Compute Pack", 34000.0, "Paid", "2026-02-25"),
    ]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?);", orders_data)

    # 3. Aggregated KPI View
    cur.execute("DROP VIEW IF EXISTS v_customer_revenue;")
    cur.execute("""
        CREATE VIEW v_customer_revenue AS
        SELECT 
            c.company_name,
            c.country,
            c.tier,
            COUNT(o.order_id) AS total_orders,
            ROUND(SUM(o.order_amount), 2) AS total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.payment_status = 'Paid'
        GROUP BY c.customer_id, c.company_name, c.country, c.tier;
    """)

    conn.commit()
    conn.close()

    return db_path, "Demo SQLite Database (Customers, Orders, & Revenue View)"
