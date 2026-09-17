"""
Base Connector Abstractions & Lifecycle Interface
Defines the standard contract for all database, cloud warehouse,
SaaS, and API data sources in the DataMind platform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional, Iterator, Union
from modules.connectors.security import CredentialVault, CredentialSanitizer, default_vault


class ConnectorType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    MONGODB = "mongodb"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    GOOGLE_SHEETS = "google_sheets"
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    REST_API = "rest_api"
    CLOUD_STORAGE = "cloud_storage"
    DATA_WAREHOUSE = "data_warehouse"


class ConnectionStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    UNTESTED = "untested"


@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: Optional[str] = None


@dataclass
class TableMetadata:
    name: str
    schema: str = "public"
    row_count_approx: Optional[int] = None
    columns: List[ColumnMetadata] = field(default_factory=list)


@dataclass
class SchemaMetadata:
    tables: List[TableMetadata] = field(default_factory=list)
    views: List[str] = field(default_factory=list)
    database_name: str = ""
    total_tables: int = 0


@dataclass
class ConnectionResult:
    success: bool
    latency_ms: float
    message: str
    server_version: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorConfig:
    connector_type: ConnectorType
    name: str
    id: str = field(default_factory=lambda: f"conn_{uuid.uuid4().hex[:8]}")
    host: str = "localhost"
    port: Optional[int] = None
    database: str = ""
    username: str = ""
    encrypted_credentials: str = ""  # Armored Fernet ciphertext
    ssl_enabled: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    status: ConnectionStatus = ConnectionStatus.UNTESTED
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_safe_dict(self) -> Dict[str, Any]:
        """Returns serialized configuration with credentials sanitized/masked."""
        d = asdict(self)
        d["encrypted_credentials"] = "[SECURE_ENCRYPTED_BLOB]" if self.encrypted_credentials else ""
        return d


class BaseConnector(ABC):
    """
    Abstract Base Class for all external data connectors.
    Enforces standardized testing, schema introspection, streaming, and execution.
    """

    def __init__(self, config: ConnectorConfig, vault: Optional[CredentialVault] = None):
        self.config = config
        self.vault = vault or default_vault
        self._active_connection = None

    def get_decrypted_credential(self) -> str:
        """Decrypts and returns the sensitive secret at connection time."""
        if not self.config.encrypted_credentials:
            return ""
        return self.vault.decrypt(self.config.encrypted_credentials)

    def set_credential(self, secret_or_payload: Union[str, dict]):
        """Encrypts and sets the credential into the configuration."""
        self.config.encrypted_credentials = self.vault.encrypt(secret_or_payload)

    @abstractmethod
    def test_connection(self) -> ConnectionResult:
        """Probes endpoint reachability, authentication, and permission sanity."""
        pass

    @abstractmethod
    def get_schema(self) -> SchemaMetadata:
        """Discovers tables, views, and column definitions."""
        pass

    @abstractmethod
    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        """Executes a parameterized SQL or native query and returns a pandas DataFrame."""
        pass

    def stream_records(self, query: str, batch_size: int = 5000) -> Iterator[pd.DataFrame]:
        """Default batch streaming fallback; subclasses can override with native cursors."""
        df = self.execute_query(query, limit=None)
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            yield df.iloc[i : i + batch_size]

    def close(self):
        """Releases all open sockets, pools, and handles."""
        if self._active_connection:
            try:
                if hasattr(self._active_connection, "close"):
                    self._active_connection.close()
            except Exception:
                pass
            finally:
                self._active_connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
