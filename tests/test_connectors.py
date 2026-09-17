"""
Unit tests for Data Connectors & Security Architecture
Verifies:
- Credential encryption and decryption with Fernet/AES
- Log and traceback secret redaction (CredentialSanitizer)
- SQLite live connection, schema discovery, and query execution
- MongoDB document flattening logic with nested BSON
- Connector factory and registry catalog integrity
"""

import os
import unittest
import sqlite3
import pandas as pd
from modules.connectors import (
    ConnectorType,
    ConnectionStatus,
    ConnectorConfig,
    CredentialVault,
    CredentialSanitizer,
    SQLiteConnector,
    flatten_mongo_documents,
    CONNECTOR_CATALOG,
    CONNECTOR_REGISTRY,
    create_connector,
)


class TestCredentialSecurity(unittest.TestCase):
    """Tests symmetric encryption and secret sanitization."""

    def setUp(self):
        self.vault = CredentialVault(master_key="Unit-Test-Secret-Key-12345")

    def test_encryption_roundtrip_string(self):
        plain = "SuperSecretP@ssw0rd!2026"
        encrypted = self.vault.encrypt(plain)
        self.assertNotEqual(plain, encrypted)
        decrypted = self.vault.decrypt(encrypted)
        self.assertEqual(plain, decrypted)

    def test_encryption_roundtrip_json(self):
        payload = {"project_id": "analytics-prod", "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgk\n-----END PRIVATE KEY-----"}
        encrypted = self.vault.encrypt(payload)
        decrypted = self.vault.decrypt_json(encrypted)
        self.assertEqual(decrypted["project_id"], "analytics-prod")
        self.assertIn("BEGIN PRIVATE KEY", decrypted["private_key"])

    def test_sanitizer_masks_connection_string(self):
        raw_uri = "postgresql+psycopg2://postgres_admin:UltraSecretPwd99@db.prod.internal:5432/analytics_db"
        sanitized = CredentialSanitizer.sanitize(raw_uri)
        self.assertNotIn("UltraSecretPwd99", sanitized)
        self.assertIn("********", sanitized)
        self.assertIn("postgres_admin", sanitized)

    def test_sanitizer_masks_bearer_token(self):
        header = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        sanitized = CredentialSanitizer.sanitize(header)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", sanitized)
        self.assertIn("[REDACTED_BEARER_TOKEN]", sanitized)

    def test_sanitizer_masks_key_values(self):
        raw_err = "OperationalError: password = my_super_secret failed for sa"
        sanitized = CredentialSanitizer.sanitize(raw_err)
        self.assertNotIn("my_super_secret", sanitized)
        self.assertIn("********", sanitized)

    def test_mask_secret_ui(self):
        masked = CredentialSanitizer.mask_secret("sk-1234567890abcdef", keep_chars=3)
        self.assertTrue(masked.startswith("sk-"))
        self.assertTrue(masked.endswith("def"))
        self.assertIn("********", masked)


class TestSQLiteLiveConnector(unittest.TestCase):
    """Tests SQLite live connection, schema introspection, and query execution."""

    def setUp(self):
        self.db_path = "test_analytics_temp.db"
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS customers;")
        cur.execute("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                revenue REAL,
                status TEXT
            );
        """)
        cur.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?);",
            [
                (101, "Acme Corp", 15400.50, "Active"),
                (102, "Beta LLC", 8200.00, "Active"),
                (103, "Gamma Tech", 31200.75, "Churned")
            ]
        )
        conn.commit()
        conn.close()

        self.config = ConnectorConfig(
            connector_type=ConnectorType.SQLITE,
            name="Test SQLite DB",
            database=self.db_path
        )
        self.connector = SQLiteConnector(self.config)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_connection_test(self):
        res = self.connector.test_connection()
        self.assertTrue(res.success)
        self.assertIn("SQLite", res.message)
        self.assertGreater(res.latency_ms, 0)
        self.assertEqual(self.config.status, ConnectionStatus.CONNECTED)

    def test_schema_discovery(self):
        schema = self.connector.get_schema()
        self.assertGreaterEqual(schema.total_tables, 1)
        table_names = [t.name for t in schema.tables]
        self.assertIn("customers", table_names)

        cust_table = next(t for t in schema.tables if t.name == "customers")
        self.assertEqual(cust_table.row_count_approx, 3)
        col_names = [c.name for c in cust_table.columns]
        self.assertIn("customer_id", col_names)
        self.assertIn("revenue", col_names)

        # Primary key verification
        pk_col = next(c for c in cust_table.columns if c.name == "customer_id")
        self.assertTrue(pk_col.is_primary_key)

    def test_query_execution(self):
        df = self.connector.execute_query("SELECT name, revenue FROM customers WHERE revenue > 10000 ORDER BY revenue DESC;")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["name"], "Gamma Tech")
        self.assertEqual(float(df.iloc[0]["revenue"]), 31200.75)


class TestMongoDBDocumentFlattening(unittest.TestCase):
    """Tests normalization of nested documents into flat DataFrames."""

    def test_flatten_nested_documents(self):
        sample_docs = [
            {
                "_id": "60d5ec49f1b2c8b1f8e4e1a1",
                "customer": "Alice",
                "metrics": {"sessions": 42, "score": 9.5},
                "tags": ["premium", "vip"]
            },
            {
                "_id": "60d5ec49f1b2c8b1f8e4e1a2",
                "customer": "Bob",
                "metrics": {"sessions": 12, "score": 7.1},
                "tags": ["standard"]
            }
        ]
        df = flatten_mongo_documents(sample_docs, max_depth=2)
        self.assertEqual(len(df), 2)
        self.assertIn("customer", df.columns)
        self.assertIn("metrics.sessions", df.columns)
        self.assertIn("metrics.score", df.columns)
        self.assertEqual(df.iloc[0]["metrics.sessions"], 42)


class TestConnectorRegistryAndCatalog(unittest.TestCase):
    """Tests factory registration and catalog coverage."""

    def test_catalog_has_all_required_sources(self):
        catalog_types = {item["type"] for item in CONNECTOR_CATALOG}
        required = {
            "postgresql", "mysql", "sqlite", "mssql", "mongodb",
            "snowflake", "bigquery", "google_sheets", "google_drive",
            "onedrive", "rest_api", "cloud_storage"
        }
        for req in required:
            self.assertIn(req, catalog_types, f"Catalog missing {req}")

    def test_factory_instantiation_sqlite(self):
        cfg = ConnectorConfig(
            connector_type=ConnectorType.SQLITE,
            name="In-Memory SQLite",
            database=":memory:"
        )
        conn = create_connector(cfg)
        self.assertIsInstance(conn, SQLiteConnector)

    def test_credential_masking_in_config(self):
        vault = CredentialVault()
        cfg = ConnectorConfig(
            connector_type=ConnectorType.POSTGRESQL,
            name="Prod Postgres",
            host="db.prod.internal",
            username="admin"
        )
        conn = SQLiteConnector(cfg, vault=vault)
        conn.set_credential("VerySecretPassword99!")
        safe_dict = cfg.to_safe_dict()
        self.assertNotIn("VerySecretPassword99!", str(safe_dict))
        self.assertEqual(safe_dict["encrypted_credentials"], "[SECURE_ENCRYPTED_BLOB]")
        self.assertEqual(conn.get_decrypted_credential(), "VerySecretPassword99!")


if __name__ == "__main__":
    unittest.main()
