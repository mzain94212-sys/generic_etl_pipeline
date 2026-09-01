"""
Core Data Schemas and Enums for the Generic ETL Pipeline.
Defines semantic data types, profiling models, transformation options, and pipeline states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SemanticType(str, Enum):
    """Semantic data types detected across any dataset."""
    # Contact & Personal
    PHONE_PAKISTAN = "phone_pakistan"
    PHONE_INTERNATIONAL = "phone_international"
    EMAIL = "email"
    CNIC_PAKISTAN = "cnic_pakistan"
    NAME_PERSON = "name_person"
    AGE = "age"
    GENDER = "gender"
    ADDRESS = "address"
    CITY = "city"
    COUNTRY = "country"
    POSTAL_CODE = "postal_code"

    # Temporal
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    YEAR = "year"
    TIMESTAMP_EPOCH = "timestamp_epoch"

    # Financial & Numeric
    CURRENCY_AMOUNT = "currency_amount"
    PERCENTAGE = "percentage"
    NUMERIC_INTEGER = "numeric_integer"
    NUMERIC_FLOAT = "numeric_float"
    IDENTIFIER_ID = "identifier_id"

    # Web & Networking
    URL = "url"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"

    # Categorical & Logical
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"

    # Text & Other
    TEXT_LONG = "text_long"
    TEXT_GENERIC = "text_generic"
    JSON_EMBEDDED = "json_embedded"
    UNKNOWN = "unknown"


class ImputationStrategy(str, Enum):
    """Missing value imputation strategies."""
    AUTO = "auto"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    CONSTANT = "constant"
    DROP_ROW = "drop_row"
    NONE = "none"


class OutlierStrategy(str, Enum):
    """Outlier handling strategies."""
    NONE = "none"
    CLIP_IQR = "clip_iqr"
    CLIP_ZSCORE = "clip_zscore"
    DROP = "drop"
    SET_NULL = "set_null"


@dataclass
class ColumnProfile:
    """Detailed profile of a single column detected in the dataset."""
    col_name: str
    original_dtype: str
    detected_type: SemanticType
    confidence_score: float  # 0.0 to 1.0
    detected_format: Optional[str] = None  # e.g., '%Y-%m-%d', '03XX-XXXXXXX'
    total_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    sample_values: List[Any] = field(default_factory=list)
    top_values: Dict[str, int] = field(default_factory=dict)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None
    outlier_count: int = 0
    regex_match_rate: float = 0.0
    col_name_similarity: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    """Overall dataset metadata and profile."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    file_type: str
    file_size_bytes: int
    duplicate_rows_count: int = 0
    memory_usage_bytes: int = 0


@dataclass
class ColumnCleanConfig:
    """User-customizable or auto-selected cleaning config per column."""
    col_name: str
    target_semantic_type: SemanticType
    imputation_strategy: ImputationStrategy = ImputationStrategy.AUTO
    imputation_constant_value: Optional[Any] = None
    outlier_strategy: OutlierStrategy = OutlierStrategy.NONE
    outlier_threshold: float = 1.5  # 1.5 * IQR or 3.0 Z-score
    standardize_phone: bool = True
    phone_default_country: str = "PK"  # Pakistan default
    phone_prefix: str = "92"  # or "+92" or "03"
    standardize_datetime: bool = True
    datetime_target_format: str = "%Y-%m-%d %H:%M:%S"
    strip_whitespace: bool = True
    normalize_casing: Optional[str] = None  # 'lower', 'upper', 'title', None
    remove_currency_symbols: bool = True
    enable_validation: bool = True


@dataclass
class DatasetCleanConfig:
    """Global cleaning and transformation parameters."""
    columns: Dict[str, ColumnCleanConfig] = field(default_factory=dict)
    drop_duplicate_rows: bool = True
    fuzzy_deduplication_key: Optional[List[str]] = None
    fuzzy_dedup_threshold: float = 0.85
    drop_columns_with_high_nulls: bool = False
    null_column_drop_threshold: float = 0.80  # drop if > 80% nulls
    drop_rows_with_high_nulls: bool = False
    null_row_drop_threshold: float = 0.60


@dataclass
class ValidationReport:
    """Results of post-transformation validation."""
    passed: bool
    total_validations: int
    valid_count: int
    invalid_count: int
    column_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)


@dataclass
class ESLoadConfig:
    """Configuration for Elasticsearch ingestion."""
    host: str = "http://localhost:9200"
    index_name: str = "generic_dataset"
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    use_ssl: bool = False
    verify_certs: bool = False
    batch_size: int = 500
    create_index_if_missing: bool = True
    overwrite_index: bool = False
    use_mock_if_unavailable: bool = True


@dataclass
class PipelineExecutionResult:
    """Full execution summary of the ETL run."""
    success: bool
    stage: str
    extracted_rows: int = 0
    extracted_cols: int = 0
    cleaned_rows: int = 0
    cleaned_cols: int = 0
    indexed_rows: int = 0
    execution_time_seconds: float = 0.0
    dataset_profile: Optional[DatasetProfile] = None
    validation_report: Optional[ValidationReport] = None
    es_mapping: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
