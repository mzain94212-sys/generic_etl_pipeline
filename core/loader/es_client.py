"""
Elasticsearch Connection Manager and High-Speed In-Memory Mock Fallback Engine.
Provides transparent connectivity to live Elasticsearch clusters or switches smoothly
to a local in-memory searchable engine when no cluster is reachable.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from ..schemas import ESLoadConfig

try:
    from elasticsearch import Elasticsearch, helpers
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False


class MockIndicesClient:
    """Mock Elasticsearch Indices API."""
    def __init__(self, mock_store: Dict[str, Dict[str, Any]]):
        self.mock_store = mock_store

    def exists(self, index: str) -> bool:
        return index in self.mock_store

    def create(self, index: str, body: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if index not in self.mock_store:
            self.mock_store[index] = {
                'mapping': body or {},
                'docs': {},
                'created_at': time.time()
            }
        return {"acknowledged": True, "shards_acknowledged": True, "index": index}

    def delete(self, index: str, **kwargs) -> Dict[str, Any]:
        if index in self.mock_store:
            del self.mock_store[index]
            return {"acknowledged": True}
        return {"acknowledged": False}

    def get_mapping(self, index: str, **kwargs) -> Dict[str, Any]:
        if index in self.mock_store:
            return {index: self.mock_store[index].get('mapping', {})}
        return {}


class MockElasticsearchClient:
    """In-memory full-featured Mock Elasticsearch client for testing and standalone operation."""
    def __init__(self):
        self._indices: Dict[str, Dict[str, Any]] = {}
        self.indices = MockIndicesClient(self._indices)

    def ping(self) -> bool:
        return True

    def info(self) -> Dict[str, Any]:
        return {
            "name": "mock-elasticsearch-node",
            "cluster_name": "in-memory-etl-cluster",
            "version": {"number": "8.14.0-mock", "build_flavor": "mock"},
            "tagline": "You Know, for Search (Mock Mode)"
        }

    def index(self, index: str, document: Dict[str, Any], id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if index not in self._indices:
            self.indices.create(index)
        doc_id = str(id) if id is not None else str(len(self._indices[index]['docs']) + 1)
        self._indices[index]['docs'][doc_id] = document
        return {"_index": index, "_id": doc_id, "result": "created"}

    def count(self, index: str, **kwargs) -> Dict[str, Any]:
        if index in self._indices:
            return {"count": len(self._indices[index]['docs'])}
        return {"count": 0}

    def search(self, index: str, query: Optional[Dict[str, Any]] = None, size: int = 50, from_: int = 0, **kwargs) -> Dict[str, Any]:
        """Perform in-memory multi-match, match, wildcard, and term search."""
        if index not in self._indices:
            return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}

        docs = self._indices[index]['docs']
        results = []

        q = (query or {}).get('query', query or {})
        
        # Check if match_all or empty query
        if not q or 'match_all' in q:
            for doc_id, source in docs.items():
                results.append({"_index": index, "_id": doc_id, "_score": 1.0, "_source": source})
        else:
            # Simple keyword / multi_match search evaluator
            search_term = ""
            if 'multi_match' in q:
                search_term = str(q['multi_match'].get('query', '')).lower()
            elif 'match' in q:
                field_key = list(q['match'].keys())[0]
                search_term = str(q['match'][field_key]).lower()
            elif 'query_string' in q:
                search_term = str(q['query_string'].get('query', '')).lower()

            for doc_id, source in docs.items():
                score = 0.0
                match = False
                if not search_term or search_term == "*":
                    match = True
                    score = 1.0
                else:
                    # Check across all fields in document
                    for k, v in source.items():
                        v_str = str(v).lower()
                        if search_term in v_str:
                            match = True
                            score += 1.0
                if match:
                    results.append({"_index": index, "_id": doc_id, "_score": score, "_source": source})

        total_hits = len(results)
        paginated_hits = results[from_:from_ + size]
        return {
            "took": 1,
            "timed_out": False,
            "hits": {
                "total": {"value": total_hits, "relation": "eq"},
                "max_score": 1.0,
                "hits": paginated_hits
            }
        }


class ESClientManager:
    """Manages Elasticsearch connections and fallbacks."""

    @classmethod
    def get_client(cls, config: ESLoadConfig) -> Tuple[Union[Any, MockElasticsearchClient], bool, str]:
        """
        Connect to real Elasticsearch or initialize mock client if unreachable.
        Returns (client, is_mock, status_message).
        """
        if HAS_ELASTICSEARCH and config.host:
            try:
                auth = None
                if config.username and config.password:
                    auth = (config.username, config.password)

                es = Elasticsearch(
                    config.host,
                    basic_auth=auth,
                    api_key=config.api_key if config.api_key else None,
                    verify_certs=config.verify_certs,
                    request_timeout=4
                )
                if es.ping():
                    info = es.info()
                    ver = info.get('version', {}).get('number', 'Unknown')
                    return es, False, f"Connected to Elasticsearch v{ver} at {config.host}"
            except Exception as e:
                pass

        # Fallback to Mock Client
        if config.use_mock_if_unavailable:
            mock_es = MockElasticsearchClient()
            return mock_es, True, "Elasticsearch cluster offline. Running in High-Speed In-Memory Mock ES Engine."
        
        raise ConnectionError(f"Failed to connect to Elasticsearch at {config.host}")
