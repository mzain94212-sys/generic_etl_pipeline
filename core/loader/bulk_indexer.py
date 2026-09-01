"""
Elasticsearch Streaming Bulk Indexer.
Performs chunked, high-throughput document ingestion with schema validation,
dynamic mapping creation, and detailed ingestion telemetry.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..schemas import ColumnCleanConfig, ESLoadConfig
from .es_client import ESClientManager, MockElasticsearchClient
from .schema_generator import DynamicESMappingGenerator

try:
    from elasticsearch import helpers
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False


class ESBulkIndexer:
    """Streams cleaned datasets directly into Elasticsearch indices."""

    @classmethod
    def index_dataframe(
        cls, 
        df: pd.DataFrame, 
        clean_configs: Dict[str, ColumnCleanConfig],
        load_config: ESLoadConfig
    ) -> Tuple[int, int, float, Dict[str, Any], str]:
        """
        Index a pandas DataFrame into Elasticsearch.
        Returns (success_count, failed_count, duration_seconds, mapping, message).
        """
        start_time = time.time()
        client, is_mock, conn_msg = ESClientManager.get_client(load_config)

        # 1. Generate dynamic mapping
        mapping = DynamicESMappingGenerator.generate_mapping(clean_configs)
        index_name = load_config.index_name.lower().strip().replace(' ', '_')

        # 2. Check and Create Index
        try:
            if client.indices.exists(index=index_name):
                if load_config.overwrite_index:
                    client.indices.delete(index=index_name)
                    client.indices.create(index=index_name, body=mapping)
            else:
                if load_config.create_index_if_missing:
                    client.indices.create(index=index_name, body=mapping)
        except Exception as e:
            # Continue if index already exists
            pass

        # 3. Sanitize DataFrame for JSON compatibility (replace NaNs with None)
        sanitized_df = df.copy()
        for c in sanitized_df.columns:
            sanitized_df[c] = sanitized_df[c].apply(lambda x: None if pd.isna(x) else x)

        records = sanitized_df.to_dict(orient='records')
        total_records = len(records)
        success_count = 0
        failed_count = 0

        # 4. Stream Ingestion
        if is_mock or not HAS_ELASTICSEARCH:
            # Use mock bulk
            for i, doc in enumerate(records):
                try:
                    client.index(index=index_name, document=doc, id=str(i + 1))
                    success_count += 1
                except Exception:
                    failed_count += 1
        else:
            # Use real Elasticsearch helpers.bulk
            def doc_generator():
                for i, doc in enumerate(records):
                    yield {
                        "_index": index_name,
                        "_id": str(i + 1),
                        "_source": doc
                    }

            try:
                for ok, info in helpers.streaming_bulk(
                    client,
                    doc_generator(),
                    chunk_size=load_config.batch_size,
                    raise_on_error=False
                ):
                    if ok:
                        success_count += 1
                    else:
                        failed_count += 1
            except Exception as e:
                # Fallback to single doc indexing
                for i, doc in enumerate(records):
                    try:
                        client.index(index=index_name, document=doc, id=str(i + 1))
                        success_count += 1
                    except Exception:
                        failed_count += 1

        duration = time.time() - start_time
        status_msg = f"Indexed {success_count}/{total_records} documents into '{index_name}' in {duration:.2f}s ({conn_msg})"

        return success_count, failed_count, round(duration, 3), mapping, status_msg
