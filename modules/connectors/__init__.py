"""
Data Connectors & Integrations Package
Enterprise connectivity for relational databases, NoSQL, data warehouses, and SaaS endpoints.
"""

from modules.connectors.base import (
    ConnectorType,
    ConnectionStatus,
    ConnectorConfig,
    ConnectionResult,
    SchemaMetadata,
    TableMetadata,
    ColumnMetadata,
    BaseConnector,
)
from modules.connectors.security import (
    CredentialVault,
    CredentialSanitizer,
    default_vault,
)
from modules.connectors.relational import (
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    MSSQLServerConnector,
)
from modules.connectors.nosql import MongoDBConnector, flatten_mongo_documents
from modules.connectors.warehouse import SnowflakeConnector, BigQueryConnector
from modules.connectors.extensible import (
    GoogleSheetsConnector,
    GoogleDriveConnector,
    OneDriveConnector,
    RESTAPIConnector,
    CloudStorageConnector,
)
from modules.connectors.registry import (
    CONNECTOR_REGISTRY,
    CONNECTOR_CATALOG,
    create_connector,
)
from modules.connectors.demo_db import get_or_create_demo_sqlite

__all__ = [
    "ConnectorType",
    "ConnectionStatus",
    "ConnectorConfig",
    "ConnectionResult",
    "SchemaMetadata",
    "TableMetadata",
    "ColumnMetadata",
    "BaseConnector",
    "CredentialVault",
    "CredentialSanitizer",
    "default_vault",
    "PostgreSQLConnector",
    "MySQLConnector",
    "SQLiteConnector",
    "MSSQLServerConnector",
    "MongoDBConnector",
    "flatten_mongo_documents",
    "SnowflakeConnector",
    "BigQueryConnector",
    "GoogleSheetsConnector",
    "GoogleDriveConnector",
    "OneDriveConnector",
    "RESTAPIConnector",
    "CloudStorageConnector",
    "CONNECTOR_REGISTRY",
    "CONNECTOR_CATALOG",
    "create_connector",
    "get_or_create_demo_sqlite",
]
