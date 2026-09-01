"""Unit tests for Elasticsearch mapping generator, Mock ES client, and bulk indexer."""

import pandas as pd
from core.schemas import ColumnCleanConfig, ESLoadConfig, SemanticType
from core.loader.schema_generator import DynamicESMappingGenerator
from core.loader.es_client import MockElasticsearchClient
from core.loader.bulk_indexer import ESBulkIndexer


def test_dynamic_es_mapping_generation():
    clean_configs = {
        "mob_number": ColumnCleanConfig("mob_number", target_semantic_type=SemanticType.PHONE_PAKISTAN),
        "created_at": ColumnCleanConfig("created_at", target_semantic_type=SemanticType.DATETIME),
        "salary": ColumnCleanConfig("salary", target_semantic_type=SemanticType.CURRENCY_AMOUNT),
        "age": ColumnCleanConfig("age", target_semantic_type=SemanticType.AGE),
        "city": ColumnCleanConfig("city", target_semantic_type=SemanticType.CITY)
    }

    mapping = DynamicESMappingGenerator.generate_mapping(clean_configs)
    props = mapping["mappings"]["properties"]

    assert props["mob_number"]["type"] == "text"
    assert "keyword" in props["mob_number"]["fields"]
    assert props["created_at"]["type"] == "date"
    assert props["salary"]["type"] == "double"
    assert props["age"]["type"] == "long"
    assert props["city"]["type"] == "keyword"


def test_mock_es_bulk_and_search():
    df = pd.DataFrame([
        {"cust_name": "Muhammad Zain", "phone": "923001234567", "city": "Lahore"},
        {"cust_name": "Ali Hassan", "phone": "923219876543", "city": "Islamabad"}
    ])

    clean_configs = {
        "cust_name": ColumnCleanConfig("cust_name", target_semantic_type=SemanticType.NAME_PERSON),
        "phone": ColumnCleanConfig("phone", target_semantic_type=SemanticType.PHONE_PAKISTAN),
        "city": ColumnCleanConfig("city", target_semantic_type=SemanticType.CITY)
    }

    load_cfg = ESLoadConfig(index_name="test_index", use_mock_if_unavailable=True)
    success, failed, duration, _, _ = ESBulkIndexer.index_dataframe(df, clean_configs, load_cfg)

    assert success == 2
    assert failed == 0

    # Search test
    client = MockElasticsearchClient()
    client.index("test_index", {"cust_name": "Muhammad Zain", "phone": "923001234567"})
    res = client.search("test_index", query={"query": {"match": {"cust_name": "Zain"}}})
    assert res["hits"]["total"]["value"] >= 1


if __name__ == "__main__":
    test_dynamic_es_mapping_generation()
    test_mock_es_bulk_and_search()
    print("All Loader & Elasticsearch tests passed successfully!")
