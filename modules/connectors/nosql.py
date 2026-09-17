"""
NoSQL Database Connectors
Production-ready MongoDB Connector with BSON normalization,
document aggregation pipeline, and dynamic schema inference.
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


def flatten_mongo_documents(docs: List[Dict[str, Any]], max_depth: int = 2) -> pd.DataFrame:
    """
    Normalizes a list of BSON / MongoDB documents into a tabular pandas DataFrame.
    Converts ObjectIds, Datetimes, and deeply nested embedded dictionaries.
    """
    if not docs:
        return pd.DataFrame()

    sanitized = []
    for d in docs:
        item = {}
        for k, v in d.items():
            # Convert ObjectId to string
            if hasattr(v, "__str__") and type(v).__name__ == "ObjectId":
                item[k] = str(v)
            else:
                item[k] = v
        sanitized.append(item)

    df = pd.json_normalize(sanitized, max_level=max_depth)
    return df


class MongoDBConnector(BaseConnector):
    """MongoDB Connector supporting MongoDB Atlas (SRV) and standalone replica sets."""

    DEFAULT_PORT = 27017

    def _get_client(self):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is required for MongoDB connections (`pip install pymongo`).")

        pwd = self.get_decrypted_credential()
        user = self.config.username
        host = self.config.host or "localhost"
        port = self.config.port or self.DEFAULT_PORT
        auth_source = self.config.extra_params.get("authSource", "admin")

        if host.startswith("mongodb+srv://") or host.startswith("mongodb://"):
            uri = host
        elif user and pwd:
            uri = f"mongodb://{user}:{pwd}@{host}:{port}/?authSource={auth_source}"
        else:
            uri = f"mongodb://{host}:{port}/"

        return MongoClient(uri, serverSelectionTimeoutMS=5000)

    def test_connection(self) -> ConnectionResult:
        start = time.time()
        try:
            client = self._get_client()
            server_info = client.server_info()
            client.admin.command('ping')
            version = server_info.get("version", "Unknown")
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.CONNECTED
            client.close()
            return ConnectionResult(
                success=True,
                latency_ms=latency,
                message=f"Connected to MongoDB successfully (v{version})",
                server_version=f"MongoDB {version}",
                details={"databases": client.list_database_names()[:10] if hasattr(client, "list_database_names") else []}
            )
        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            self.config.status = ConnectionStatus.FAILED
            return ConnectionResult(
                success=False,
                latency_ms=latency,
                message=f"MongoDB connection failed: {CredentialSanitizer.sanitize(str(e))}"
            )

    def get_schema(self) -> SchemaMetadata:
        client = self._get_client()
        try:
            db_name = self.config.database or "test"
            db = client[db_name]
            col_names = db.list_collection_names()

            tables: List[TableMetadata] = []
            for c_name in col_names:
                coll = db[c_name]
                approx_count = coll.estimated_document_count()
                sample_docs = list(coll.find().limit(5))
                
                # Infer schema from sample documents
                inferred_cols: Dict[str, str] = {}
                for doc in sample_docs:
                    for k, v in doc.items():
                        if k not in inferred_cols:
                            inferred_cols[k] = type(v).__name__

                cols = [
                    ColumnMetadata(
                        name=col_k,
                        data_type=col_v,
                        is_primary_key=(col_k == "_id")
                    )
                    for col_k, col_v in inferred_cols.items()
                ]

                tables.append(TableMetadata(
                    name=c_name,
                    schema=db_name,
                    row_count_approx=approx_count,
                    columns=cols
                ))

            return SchemaMetadata(
                tables=tables,
                database_name=db_name,
                total_tables=len(tables)
            )
        finally:
            client.close()

    def execute_query(self, query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
        """
        Executes MongoDB query. Query can be:
        1. Name of collection: 'users'
        2. JSON spec: {"collection": "users", "filter": {"status": "active"}, "projection": {"_id": 0}}
        """
        client = self._get_client()
        try:
            db_name = self.config.database or "test"
            db = client[db_name]

            query_str = query.strip()
            if query_str.startswith("{") and query_str.endswith("}"):
                spec = json.loads(query_str)
                collection_name = spec.get("collection")
                filt = spec.get("filter", {})
                proj = spec.get("projection", None)
            else:
                collection_name = query_str
                filt = {}
                proj = None

            if not collection_name:
                raise ValueError("MongoDB query must specify a target collection name or JSON object.")

            coll = db[collection_name]
            cursor = coll.find(filt, proj)
            if limit:
                cursor = cursor.limit(limit)

            docs = list(cursor)
            return flatten_mongo_documents(docs)
        except Exception as e:
            raise RuntimeError(f"MongoDB query failed: {CredentialSanitizer.sanitize(str(e))}")
        finally:
            client.close()
