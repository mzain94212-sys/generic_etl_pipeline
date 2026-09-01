"""
Dynamic Elasticsearch Schema and Mapping Generator.
Translates detected generic semantic types into optimized Elasticsearch index mappings,
including text analyzers, keyword subfields, date formats, and numeric types.
"""

from typing import Any, Dict
from ..schemas import ColumnCleanConfig, SemanticType


class DynamicESMappingGenerator:
    """Generates Elasticsearch index creation settings and field mappings."""

    @classmethod
    def generate_mapping(cls, clean_configs: Dict[str, ColumnCleanConfig]) -> Dict[str, Any]:
        """
        Generate Elasticsearch 8.x / 9.x compatible mapping from column configurations.
        """
        properties: Dict[str, Any] = {}

        for col, cfg in clean_configs.items():
            sem_type = cfg.target_semantic_type

            if sem_type in [SemanticType.PHONE_PAKISTAN, SemanticType.PHONE_INTERNATIONAL, 
                            SemanticType.CNIC_PAKISTAN, SemanticType.EMAIL, 
                            SemanticType.NAME_PERSON, SemanticType.ADDRESS, 
                            SemanticType.TEXT_GENERIC, SemanticType.URL]:
                properties[col] = {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }

            elif sem_type in [SemanticType.DATETIME, SemanticType.DATE]:
                properties[col] = {
                    "type": "date",
                    "format": "strict_date_optional_time||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                }

            elif sem_type in [SemanticType.NUMERIC_INTEGER, SemanticType.AGE, 
                            SemanticType.IDENTIFIER_ID, SemanticType.YEAR]:
                properties[col] = {
                    "type": "long"
                }

            elif sem_type in [SemanticType.NUMERIC_FLOAT, SemanticType.CURRENCY_AMOUNT, 
                            SemanticType.PERCENTAGE]:
                properties[col] = {
                    "type": "double"
                }

            elif sem_type == SemanticType.BOOLEAN:
                properties[col] = {
                    "type": "boolean"
                }

            elif sem_type == SemanticType.IP_ADDRESS:
                properties[col] = {
                    "type": "ip"
                }

            elif sem_type in [SemanticType.CATEGORICAL, SemanticType.GENDER, 
                            SemanticType.CITY, SemanticType.COUNTRY, 
                            SemanticType.POSTAL_CODE]:
                properties[col] = {
                    "type": "keyword"
                }

            elif sem_type == SemanticType.TEXT_LONG:
                properties[col] = {
                    "type": "text",
                    "analyzer": "standard"
                }

            else:
                properties[col] = {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }

        mapping_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "properties": properties
            }
        }
        return mapping_body
