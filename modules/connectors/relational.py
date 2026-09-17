"""
Relational Database Connectors
Production-grade connectors for PostgreSQL, MySQL, SQLite, and Microsoft SQL Server.
Supports connection pooling, secure credential extraction, schema discovery,
and DataFrame query extraction with graceful driver fallback.
"""

import os
import time
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional, Iterator
from modules.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectionResult,
    SchemaMetadata,
    TableMetadata,
    ColumnMetadata,
    ConnectionStatus,
)
from modules.connectors.security import CredentialSanitizer


# ==============================================================================
# 1. SQLITE CONNECTOR
# ==============================================================================
class SQLiteConnector(BaseConnector):
    """Zero-dependency embedded relational database connector using standard sqlite3."""

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Default database to path
        self.db_path = self.config.database or self.config.host or ":memory:"

    def _get_connection(self):
        # Path safety check
        if self.db_path != ":memory:" and not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database file not found at: {self.db_path}")
        return sqlite3.connect(self.db_path, timeout=10.0)

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            ver = cursor.fetchone()[0]
            conn.close()
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message=f"Connected to SQLite successfully (v{ver})",
                server_version=f"SQLite {ver}",
                details={"file_path": self.db_path}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"SQLite connection error: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [row[0] for row in cursor.fetchall()]

            tables = []
            for t_name in table_names:
                cursor.execute(f"PRAGMA table_info('{t_name}');")
                col_rows = cursor.fetchall()
                cols = [
                    ColumnMetadata(
                        name=c[1],
                        data_type=c[2] or "TEXT",
                        nullable=not bool(c[3]),
                        is_primary_key=bool(c[5])
                    )
                    for c in col_rows
                ]
                # Row count estimate
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM '{t_name}';")
                    cnt = cursor.fetchone()[0]
                except Exception:
                    cnt = None

                tables.append(TableMetadata(name=t_name, schema="main", row_count_approx=cnt, columns=cols))

            cursor.execute("SELECT name FROM sqlite_master WHERE type='view';")
            views = [row[0] for row in cursor.fetchall()]

            return SchemaMetadata(
                tables=tables,
                views=views,
                database_name=os.path.basename(self.db_path) if self.db_path != ":memory:" else "memory",
                total_tables=len(tables)
            )
        finally:
            conn.close()

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        conn = self._get_connection()
        try:
            trimmed = query.strip().rstrip(";")
            if limit and "LIMIT" not in trimmed.upper():
                trimmed = f"{trimmed} LIMIT {limit}"
            return pd.read_sql_query(trimmed, conn)
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {CredentialSanitizer.sanitize(str(e))}")
        finally:
            conn.close()


# ==============================================================================
# 2. POSTGRESQL CONNECTOR
# ==============================================================================
class PostgreSQLConnector(BaseConnector):
    """PostgreSQL Connector using SQLAlchemy/psycopg2 with SSL and streaming cursor support."""

    DEFAULT_PORT = 5432

    def _get_engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("SQLAlchemy is required for PostgreSQL connections (`pip install psycopg2-binary sqlalchemy`).")

        pwd = self.get_decrypted_credential()
        user = self.config.username or "postgres"
        host = self.config.host or "localhost"
        port = self.config.port or self.DEFAULT_PORT
        db = self.config.database or "postgres"
        sslmode = "require" if self.config.ssl_enabled else "disable"

        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode={sslmode}"
        return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                res = conn.execute("SELECT version();").scalar()
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message="Successfully connected to PostgreSQL cluster",
                server_version=str(res)[:60],
                details={"host": self.config.host, "database": self.config.database}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"PostgreSQL connection failed: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        engine = self._get_engine()
        query = """
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position;
        """
        df = pd.read_sql_query(query, engine)
        tables_dict: Dict[str, TableMetadata] = {}

        for _, row in df.iterrows():
            key = f"{row['table_schema']}.{row['table_name']}"
            if key not in tables_dict:
                tables_dict[key] = TableMetadata(
                    name=row["table_name"],
                    schema=row["table_schema"],
                    columns=[]
                )
            tables_dict[key].columns.append(
                ColumnMetadata(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    nullable=(row["is_nullable"] == "YES")
                )
            )

        return SchemaMetadata(
            tables=list(tables_dict.values()),
            database_name=self.config.database,
            total_tables=len(tables_dict)
        )

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        engine = self._get_engine()
        trimmed = query.strip().rstrip(";")
        if limit and "LIMIT" not in trimmed.upper():
            trimmed = f"{trimmed} LIMIT {limit}"
        try:
            return pd.read_sql_query(trimmed, engine)
        except Exception as e:
            raise RuntimeError(f"PostgreSQL query failed: {CredentialSanitizer.sanitize(str(e))}")


# ==============================================================================
# 3. MYSQL CONNECTOR
# ==============================================================================
class MySQLConnector(BaseConnector):
    """MySQL / MariaDB Connector using pymysql or mysql-connector-python."""

    DEFAULT_PORT = 3306

    def _get_engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("SQLAlchemy and pymysql/mysql-connector are required (`pip install pymysql sqlalchemy`).")

        pwd = self.get_decrypted_credential()
        user = self.config.username or "root"
        host = self.config.host or "localhost"
        port = self.config.port or self.DEFAULT_PORT
        db = self.config.database or ""

        url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
        return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                res = conn.execute("SELECT VERSION();").scalar()
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message="Successfully connected to MySQL database",
                server_version=f"MySQL {res}",
                details={"host": self.config.host, "database": self.config.database}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"MySQL connection error: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        engine = self._get_engine()
        query = f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.config.database}'
            ORDER BY TABLE_NAME, ORDINAL_POSITION;
        """
        df = pd.read_sql_query(query, engine)
        tables_dict: Dict[str, TableMetadata] = {}

        for _, row in df.iterrows():
            t_name = row["TABLE_NAME"]
            if t_name not in tables_dict:
                tables_dict[t_name] = TableMetadata(
                    name=t_name,
                    schema=row["TABLE_SCHEMA"],
                    columns=[]
                )
            tables_dict[t_name].columns.append(
                ColumnMetadata(
                    name=row["COLUMN_NAME"],
                    data_type=row["DATA_TYPE"],
                    nullable=(row["IS_NULLABLE"] == "YES"),
                    is_primary_key=(row["COLUMN_KEY"] == "PRI")
                )
            )

        return SchemaMetadata(
            tables=list(tables_dict.values()),
            database_name=self.config.database,
            total_tables=len(tables_dict)
        )

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        engine = self._get_engine()
        trimmed = query.strip().rstrip(";")
        if limit and "LIMIT" not in trimmed.upper():
            trimmed = f"{trimmed} LIMIT {limit}"
        try:
            return pd.read_sql_query(trimmed, engine)
        except Exception as e:
            raise RuntimeError(f"MySQL query failed: {CredentialSanitizer.sanitize(str(e))}")


# ==============================================================================
# 4. MICROSOFT SQL SERVER CONNECTOR
# ==============================================================================
class MSSQLServerConnector(BaseConnector):
    """Microsoft SQL Server connector using pyodbc / pymssql with TDS driver."""

    DEFAULT_PORT = 1433

    def _get_engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("SQLAlchemy and pyodbc or pymssql are required (`pip install pyodbc sqlalchemy`).")

        pwd = self.get_decrypted_credential()
        user = self.config.username or "sa"
        host = self.config.host or "localhost"
        port = self.config.port or self.DEFAULT_PORT
        db = self.config.database or "master"
        driver = self.config.extra_params.get("driver", "ODBC Driver 18 for SQL Server")
        trust_cert = "yes" if not self.config.ssl_enabled else "no"

        # ODBC connection string
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={host},{port};DATABASE={db};"
            f"UID={user};PWD={pwd};Encrypt=yes;TrustServerCertificate={trust_cert}"
        )
        url = f"mssql+pyodbc:///?odbc_connect={conn_str}"
        return create_engine(url, pool_pre_ping=True)

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                res = conn.execute("SELECT @@VERSION;").scalar()
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message="Connected to Microsoft SQL Server",
                server_version=str(res)[:60],
                details={"server": self.config.host, "database": self.config.database}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"MS SQL Server connection failed: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        engine = self._get_engine()
        query = """
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
        """
        df = pd.read_sql_query(query, engine)
        tables_dict: Dict[str, TableMetadata] = {}

        for _, row in df.iterrows():
            key = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
            if key not in tables_dict:
                tables_dict[key] = TableMetadata(
                    name=row["TABLE_NAME"],
                    schema=row["TABLE_SCHEMA"],
                    columns=[]
                )
            tables_dict[key].columns.append(
                ColumnMetadata(
                    name=row["COLUMN_NAME"],
                    data_type=row["DATA_TYPE"],
                    nullable=(row["IS_NULLABLE"] == "YES")
                )
            )

        return SchemaMetadata(
            tables=list(tables_dict.values()),
            database_name=self.config.database,
            total_tables=len(tables_dict)
        )

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        engine = self._get_engine()
        trimmed = query.strip().rstrip(";")
        if limit and "TOP" not in trimmed.upper() and trimmed.upper().startswith("SELECT"):
            trimmed = f"SELECT TOP {limit} " + trimmed[6:]
        try:
            return pd.read_sql_query(trimmed, engine)
        except Exception as e:
            raise RuntimeError(f"MS SQL Server query failed: {CredentialSanitizer.sanitize(str(e))}")
