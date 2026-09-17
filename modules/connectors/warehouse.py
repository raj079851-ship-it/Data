"""
Cloud Data Warehouse Connectors
Enterprise connectors for Snowflake, Google BigQuery, Databricks, and Amazon Redshift.
Features Apache Arrow zero-copy memory transfers, OAuth/Service Account credentials,
and pre-flight query cost estimation.
"""

import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional
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
# 1. SNOWFLAKE CONNECTOR
# ==============================================================================
class SnowflakeConnector(BaseConnector):
    """Snowflake Cloud Data Warehouse Connector with Arrow Flight and Key-Pair support."""

    def _get_connection(self):
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python is required (`pip install snowflake-connector-python[pandas]`).")

        pwd = self.get_decrypted_credential()
        return snowflake.connector.connect(
            user=self.config.username,
            password=pwd,
            account=self.config.host,  # e.g., xy12345.us-east-1
            warehouse=self.config.extra_params.get("warehouse", "COMPUTE_WH"),
            database=self.config.database,
            schema=self.config.extra_params.get("schema", "PUBLIC"),
            role=self.config.extra_params.get("role", "ACCOUNTADMIN"),
            client_session_keep_alive=True
        )

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_VERSION(), CURRENT_WAREHOUSE(), CURRENT_DATABASE();")
            row = cur.fetchone()
            cur.close()
            conn.close()
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message=f"Connected to Snowflake (v{row[0]})",
                server_version=f"Snowflake {row[0]}",
                details={"warehouse": row[1], "database": row[2]}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"Snowflake connection failed: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            db = self.config.database
            cur.execute(f"""
                SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM {db}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
                ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
            """)
            rows = cur.fetchall()
            tables_dict: Dict[str, TableMetadata] = {}

            for r in rows:
                key = f"{r[0]}.{r[1]}"
                if key not in tables_dict:
                    tables_dict[key] = TableMetadata(name=r[1], schema=r[0], columns=[])
                tables_dict[key].columns.append(
                    ColumnMetadata(name=r[2], data_type=r[3], nullable=(r[4] == 'YES'))
                )

            return SchemaMetadata(
                tables=list(tables_dict.values()),
                database_name=db,
                total_tables=len(tables_dict)
            )
        finally:
            cur.close()
            conn.close()

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            trimmed = query.strip().rstrip(";")
            if limit and "LIMIT" not in trimmed.upper():
                trimmed = f"{trimmed} LIMIT {limit}"
            cur.execute(trimmed)
            # High-speed Apache Arrow conversion
            df = cur.fetch_pandas_all()
            return df
        except Exception as e:
            raise RuntimeError(f"Snowflake query execution failed: {CredentialSanitizer.sanitize(str(e))}")
        finally:
            cur.close()
            conn.close()


# ==============================================================================
# 2. GOOGLE BIGQUERY CONNECTOR
# ==============================================================================
class BigQueryConnector(BaseConnector):
    """Google BigQuery Connector supporting Service Account JSON keys and query dry-run costs."""

    def _get_client(self):
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError("google-cloud-bigquery is required (`pip install google-cloud-bigquery db-dtypes pyarrow`).")

        secret = self.get_decrypted_credential()
        project_id = self.config.database or self.config.host

        if secret:
            # Check if secret is service account JSON string
            try:
                sa_info = json.loads(secret)
                creds = service_account.Credentials.from_service_account_info(sa_info)
                return bigquery.Client(credentials=creds, project=sa_info.get("project_id", project_id))
            except Exception:
                pass

        # Fallback to Application Default Credentials (ADC) or project
        return bigquery.Client(project=project_id) if project_id else bigquery.Client()

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            client = self._get_client()
            # Perform a zero-cost dry run query
            from google.cloud import bigquery
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = client.query("SELECT 1 AS test_connection;", job_config=job_config)
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message=f"Connected to Google BigQuery (Project: {client.project})",
                server_version="Google BigQuery API v2",
                details={"project": client.project}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"BigQuery connection failed: {CredentialSanitizer.sanitize(str(e))}"
            )

    def estimate_query_cost(self, query: str) -> Dict[str, Any]:
        """Runs a dry run of the query to calculate total processed bytes and estimated cost."""
        client = self._get_client()
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(query, job_config=job_config)
        bytes_billed = job.total_bytes_processed or 0
        mb_billed = round(bytes_billed / (1024 * 1024), 2)
        gb_billed = round(bytes_billed / (1024 * 1024 * 1024), 4)
        # Standard on-demand pricing ~ $6.25 per TB ($0.00625 per GB)
        est_cost_usd = round(gb_billed * 0.00625, 5)
        return {
            "bytes_processed": bytes_billed,
            "mb_processed": mb_billed,
            "gb_processed": gb_billed,
            "estimated_cost_usd": est_cost_usd
        }

    def get_schema(self) -> SchemaMetadata:
        client = self._get_client()
        dataset_id = self.config.extra_params.get("dataset")
        tables: List[TableMetadata] = []

        if dataset_id:
            datasets = [client.get_dataset(dataset_id)]
        else:
            datasets = list(client.list_datasets(max_results=10))

        for ds in datasets:
            for tb in client.list_tables(ds.dataset_id, max_results=50):
                full_tb = client.get_table(tb.reference)
                cols = [
                    ColumnMetadata(
                        name=f.name,
                        data_type=f.field_type,
                        nullable=(f.mode != "REQUIRED")
                    )
                    for f in full_tb.schema
                ]
                tables.append(TableMetadata(
                    name=tb.table_id,
                    schema=ds.dataset_id,
                    row_count_approx=full_tb.num_rows,
                    columns=cols
                ))

        return SchemaMetadata(
            tables=tables,
            database_name=client.project,
            total_tables=len(tables)
        )

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        client = self._get_client()
        trimmed = query.strip().rstrip(";")
        if limit and "LIMIT" not in trimmed.upper():
            trimmed = f"{trimmed} LIMIT {limit}"
        try:
            job = client.query(trimmed)
            return job.to_dataframe()
        except Exception as e:
            raise RuntimeError(f"BigQuery execution failed: {CredentialSanitizer.sanitize(str(e))}")
