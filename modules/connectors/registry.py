"""
Connector Registry & Catalog
Provides discovery, factory instantiation, and UI catalog metadata
for all database, data warehouse, and SaaS connectors.
"""

from typing import Dict, Any, Type, List, Optional
from modules.connectors.base import ConnectorType, BaseConnector, ConnectorConfig
from modules.connectors.relational import (
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    MSSQLServerConnector,
)
from modules.connectors.nosql import MongoDBConnector
from modules.connectors.warehouse import SnowflakeConnector, BigQueryConnector
from modules.connectors.extensible import (
    GoogleSheetsConnector,
    GoogleDriveConnector,
    OneDriveConnector,
    RESTAPIConnector,
    CloudStorageConnector,
)


CONNECTOR_REGISTRY: Dict[ConnectorType, Type[BaseConnector]] = {
    ConnectorType.POSTGRESQL: PostgreSQLConnector,
    ConnectorType.MYSQL: MySQLConnector,
    ConnectorType.SQLITE: SQLiteConnector,
    ConnectorType.MSSQL: MSSQLServerConnector,
    ConnectorType.MONGODB: MongoDBConnector,
    ConnectorType.SNOWFLAKE: SnowflakeConnector,
    ConnectorType.BIGQUERY: BigQueryConnector,
    ConnectorType.GOOGLE_SHEETS: GoogleSheetsConnector,
    ConnectorType.GOOGLE_DRIVE: GoogleDriveConnector,
    ConnectorType.ONEDRIVE: OneDriveConnector,
    ConnectorType.REST_API: RESTAPIConnector,
    ConnectorType.CLOUD_STORAGE: CloudStorageConnector,
}


CONNECTOR_CATALOG: List[Dict[str, Any]] = [
    # 1. Relational Databases
    {
        "type": ConnectorType.POSTGRESQL.value,
        "name": "PostgreSQL",
        "category": "Relational Databases",
        "icon": "🐘",
        "default_port": 5432,
        "credential_type": "password",
        "credential_label": "Password",
        "host_placeholder": "db.production.internal or localhost",
        "desc": "Connect to PostgreSQL clusters with SSL and streaming cursor support."
    },
    {
        "type": ConnectorType.MYSQL.value,
        "name": "MySQL / MariaDB",
        "category": "Relational Databases",
        "icon": "🐬",
        "default_port": 3306,
        "credential_type": "password",
        "credential_label": "Password",
        "host_placeholder": "mysql.company.com or 127.0.0.1",
        "desc": "High-speed connection to MySQL and MariaDB databases."
    },
    {
        "type": ConnectorType.SQLITE.value,
        "name": "SQLite",
        "category": "Relational Databases",
        "icon": "🗄️",
        "default_port": None,
        "credential_type": "none",
        "credential_label": "Not Required",
        "host_placeholder": "path/to/database.db or :memory:",
        "desc": "Zero-dependency embedded SQL engine for local and file-based data."
    },
    {
        "type": ConnectorType.MSSQL.value,
        "name": "Microsoft SQL Server",
        "category": "Relational Databases",
        "icon": "🏢",
        "default_port": 1433,
        "credential_type": "password",
        "credential_label": "SQL Server / AD Password",
        "host_placeholder": "sqlserver.corp.local",
        "desc": "Enterprise connection via TDS protocol, Windows AD, or SQL Auth."
    },

    # 2. NoSQL & Document Stores
    {
        "type": ConnectorType.MONGODB.value,
        "name": "MongoDB",
        "category": "NoSQL Databases",
        "icon": "🍃",
        "default_port": 27017,
        "credential_type": "password",
        "credential_label": "Auth Password / Token",
        "host_placeholder": "cluster0.mongodb.net or localhost",
        "desc": "Document database with dynamic schema inference and BSON flattening."
    },

    # 3. Cloud Data Warehouses
    {
        "type": ConnectorType.SNOWFLAKE.value,
        "name": "Snowflake",
        "category": "Cloud Data Warehouses",
        "icon": "❄️",
        "default_port": 443,
        "credential_type": "password",
        "credential_label": "Password or Private Key",
        "host_placeholder": "xy12345.us-east-1.snowflakecomputing.com",
        "desc": "Elastic cloud data warehouse with high-throughput Apache Arrow transfers."
    },
    {
        "type": ConnectorType.BIGQUERY.value,
        "name": "Google BigQuery",
        "category": "Cloud Data Warehouses",
        "icon": "🔍",
        "default_port": 443,
        "credential_type": "service_account_json",
        "credential_label": "Service Account JSON Key",
        "host_placeholder": "GCP Project ID (e.g., analytics-prod-2026)",
        "desc": "Serverless analytics with dry-run query cost estimation."
    },

    # 4. SaaS & Extensible Connectors
    {
        "type": ConnectorType.GOOGLE_SHEETS.value,
        "name": "Google Sheets",
        "category": "SaaS & Files",
        "icon": "📊",
        "default_port": None,
        "credential_type": "optional_service_account",
        "credential_label": "Service Account JSON (Optional for private sheets)",
        "host_placeholder": "Spreadsheet ID or Google Sheet Share URL",
        "desc": "Real-time sync from public or enterprise private Google Spreadsheets."
    },
    {
        "type": ConnectorType.GOOGLE_DRIVE.value,
        "name": "Google Drive",
        "category": "SaaS & Files",
        "icon": "📁",
        "default_port": None,
        "credential_type": "oauth_or_link",
        "credential_label": "API Key / OAuth Token",
        "host_placeholder": "Google Drive File Share URL",
        "desc": "Load CSV, Excel, or Parquet datasets stored in Google Drive."
    },
    {
        "type": ConnectorType.ONEDRIVE.value,
        "name": "Microsoft OneDrive / SharePoint",
        "category": "SaaS & Files",
        "icon": "☁️",
        "default_port": None,
        "credential_type": "token",
        "credential_label": "Graph API Token",
        "host_placeholder": "OneDrive / SharePoint Shared Document URL",
        "desc": "Direct integration with Office 365 and SharePoint document libraries."
    },
    {
        "type": ConnectorType.REST_API.value,
        "name": "REST / GraphQL API",
        "category": "APIs & Webhooks",
        "icon": "🌐",
        "default_port": None,
        "credential_type": "bearer_or_api_key",
        "credential_label": "Bearer Token or API Key",
        "host_placeholder": "https://api.example.com/v1/data",
        "desc": "Universal HTTP endpoint crawler with pagination and JSON flattening."
    },
    {
        "type": ConnectorType.CLOUD_STORAGE.value,
        "name": "Cloud Storage (S3 / GCS / Azure)",
        "category": "Cloud Storage Lakes",
        "icon": "🪣",
        "default_port": None,
        "credential_type": "json_keys",
        "credential_label": "Storage Access Keys (JSON)",
        "host_placeholder": "s3://my-bucket/data.parquet or gs://lake/file.csv",
        "desc": "Stream and query remote cloud object stores directly into memory."
    }
]


def create_connector(config: ConnectorConfig) -> BaseConnector:
    """Factory function: instantiates the appropriate connector given a configuration."""
    cls = CONNECTOR_REGISTRY.get(config.connector_type)
    if not cls:
        raise ValueError(f"Unsupported connector type: '{config.connector_type}'")
    return cls(config)
