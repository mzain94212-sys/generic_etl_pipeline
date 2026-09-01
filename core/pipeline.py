"""
Generic ETL Pipeline Orchestrator.
Integrates Multi-Format Extraction, Semantic Profiling & Type Inference,
ML-Grade Data Transformation & Quality Validation, and Dynamic Elasticsearch Ingestion.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from .schemas import (
    ColumnCleanConfig,
    DatasetCleanConfig,
    DatasetProfile,
    ESLoadConfig,
    ImputationStrategy,
    OutlierStrategy,
    PipelineExecutionResult,
    SemanticType,
    ValidationReport,
)
from .extractor import GenericExtractor
from .detector.type_inference import GenericTypeDetector
from .transformer.normalizer import GenericNormalizer
from .transformer.missing_handler import MissingValueHandler
from .transformer.outlier_handler import OutlierHandler
from .transformer.deduplicator import FuzzyDeduplicator
from .transformer.validator import DataQualityValidator
from .loader.bulk_indexer import ESBulkIndexer


class GenericETLPipeline:
    """Universal, self-configuring ETL Pipeline for any tabular or semi-structured dataset."""

    def __init__(self):
        self.detector = GenericTypeDetector()
        self.extractor = GenericExtractor()

    def generate_default_cleaning_config(self, profile: DatasetProfile) -> DatasetCleanConfig:
        """Create intelligent default cleaning configurations based on detected semantic types."""
        col_configs: Dict[str, ColumnCleanConfig] = {}

        for col, col_prof in profile.columns.items():
            sem_type = col_prof.detected_type
            
            # Default imputation rule
            if sem_type in [SemanticType.NUMERIC_FLOAT, SemanticType.CURRENCY_AMOUNT, SemanticType.PERCENTAGE]:
                imp_strat = ImputationStrategy.MEDIAN
                out_strat = OutlierStrategy.CLIP_IQR
            elif sem_type in [SemanticType.NUMERIC_INTEGER, SemanticType.AGE]:
                imp_strat = ImputationStrategy.MEDIAN
                out_strat = OutlierStrategy.CLIP_IQR
            elif sem_type in [SemanticType.CATEGORICAL, SemanticType.GENDER, SemanticType.BOOLEAN, SemanticType.CITY, SemanticType.COUNTRY]:
                imp_strat = ImputationStrategy.MODE
                out_strat = OutlierStrategy.NONE
            elif sem_type in [SemanticType.DATETIME, SemanticType.DATE]:
                imp_strat = ImputationStrategy.FORWARD_FILL
                out_strat = OutlierStrategy.NONE
            else:
                imp_strat = ImputationStrategy.CONSTANT
                out_strat = OutlierStrategy.NONE

            # Default Pakistani phone standardization
            is_phone = sem_type in [SemanticType.PHONE_PAKISTAN, SemanticType.PHONE_INTERNATIONAL]

            col_configs[col] = ColumnCleanConfig(
                col_name=col,
                target_semantic_type=sem_type,
                imputation_strategy=imp_strat,
                outlier_strategy=out_strat,
                standardize_phone=is_phone,
                phone_prefix="92",  # Default: 923XXXXXXXXX
                standardize_datetime=sem_type in [SemanticType.DATETIME, SemanticType.DATE],
                strip_whitespace=True,
                remove_currency_symbols=sem_type == SemanticType.CURRENCY_AMOUNT
            )

        return DatasetCleanConfig(
            columns=col_configs,
            drop_duplicate_rows=True,
            drop_columns_with_high_nulls=False,
            drop_rows_with_high_nulls=False
        )

    def extract(
        self, 
        source: Union[str, Any], 
        filename: Optional[str] = None, 
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 1: Extract data from source file or stream."""
        return self.extractor.extract(source, filename=filename, **kwargs)

    def profile_and_detect(
        self, 
        df: pd.DataFrame, 
        file_type: str = "csv", 
        file_size_bytes: int = 0
    ) -> DatasetProfile:
        """Stage 2: Detect semantic types and profile dataset."""
        return self.detector.profile_dataset(df, file_type=file_type, file_size_bytes=file_size_bytes)

    def clean_and_transform(
        self, 
        df: pd.DataFrame, 
        clean_config: DatasetCleanConfig
    ) -> Tuple[pd.DataFrame, ValidationReport, Dict[str, Any]]:
        """Stage 3: ML-grade data cleaning, normalization, and quality validation."""
        df_transformed = df.copy()
        transform_logs = []

        # 1. Column Normalization & Standardization
        for col, cfg in clean_config.columns.items():
            if col not in df_transformed.columns:
                continue

            sem_type = cfg.target_semantic_type
            
            # Text / Whitespace cleaning
            if cfg.strip_whitespace or cfg.normalize_casing:
                df_transformed[col] = df_transformed[col].apply(
                    lambda v: GenericNormalizer.clean_text(v, casing=cfg.normalize_casing, strip_whitespace=cfg.strip_whitespace)
                )

            # Pakistani Phone Normalization
            if sem_type == SemanticType.PHONE_PAKISTAN and cfg.standardize_phone:
                df_transformed[col] = df_transformed[col].apply(
                    lambda v: GenericNormalizer.clean_pakistan_phone(v, prefix=cfg.phone_prefix)
                )
                transform_logs.append(f"Standardized Pakistani phone column '{col}' with prefix '{cfg.phone_prefix}'")

            # International Phone Normalization
            elif sem_type == SemanticType.PHONE_INTERNATIONAL and cfg.standardize_phone:
                df_transformed[col] = df_transformed[col].apply(
                    lambda v: GenericNormalizer.clean_international_phone(v, default_region=cfg.phone_default_country)
                )

            # CNIC Normalization
            elif sem_type == SemanticType.CNIC_PAKISTAN:
                df_transformed[col] = df_transformed[col].apply(GenericNormalizer.clean_pakistan_cnic)

            # DateTime Normalization
            elif sem_type in [SemanticType.DATETIME, SemanticType.DATE] and cfg.standardize_datetime:
                df_transformed[col] = df_transformed[col].apply(
                    lambda v: GenericNormalizer.clean_datetime(v, target_format=cfg.datetime_target_format)
                )
                transform_logs.append(f"Standardized datetime column '{col}' to format '{cfg.datetime_target_format}'")

            # Currency Normalization
            elif sem_type == SemanticType.CURRENCY_AMOUNT and cfg.remove_currency_symbols:
                df_transformed[col] = df_transformed[col].apply(GenericNormalizer.clean_currency)

            # Boolean Normalization
            elif sem_type == SemanticType.BOOLEAN:
                df_transformed[col] = df_transformed[col].apply(GenericNormalizer.clean_boolean)

        # 2. Outlier Treatment
        df_transformed, outlier_stats = OutlierHandler.handle_outliers(df_transformed, clean_config.columns)
        if outlier_stats.get('total_outliers_treated', 0) > 0:
            transform_logs.append(f"Treated {outlier_stats['total_outliers_treated']} statistical outliers")

        # 3. Missing Value Imputation
        df_transformed, imp_stats = MissingValueHandler.apply_imputation(
            df_transformed, 
            clean_config.columns, 
            global_config=clean_config
        )
        if imp_stats.get('imputed_cells', 0) > 0:
            transform_logs.append(f"Imputed {imp_stats['imputed_cells']} missing values")

        # 4. Deduplication
        df_transformed, dedup_stats = FuzzyDeduplicator.deduplicate(
            df_transformed,
            exact_dedup=clean_config.drop_duplicate_rows,
            fuzzy_keys=clean_config.fuzzy_deduplication_key,
            similarity_threshold=clean_config.fuzzy_dedup_threshold
        )
        if dedup_stats.get('exact_duplicates_removed', 0) > 0 or dedup_stats.get('fuzzy_duplicates_removed', 0) > 0:
            transform_logs.append(f"Removed {dedup_stats.get('exact_duplicates_removed', 0)} exact and {dedup_stats.get('fuzzy_duplicates_removed', 0)} fuzzy duplicate records")

        # 5. Validation Check
        validation_report = DataQualityValidator.validate_dataset(df_transformed, clean_config.columns)

        audit_summary = {
            'logs': transform_logs,
            'outlier_stats': outlier_stats,
            'imputation_stats': imp_stats,
            'dedup_stats': dedup_stats
        }

        return df_transformed, validation_report, audit_summary

    def load_to_elasticsearch(
        self, 
        df: pd.DataFrame, 
        clean_config: DatasetCleanConfig, 
        load_config: ESLoadConfig
    ) -> Tuple[int, int, float, Dict[str, Any], str]:
        """Stage 4: Dynamically generate mapping and bulk load into Elasticsearch."""
        return ESBulkIndexer.index_dataframe(df, clean_config.columns, load_config)

    def run(
        self, 
        source: Union[str, Any], 
        filename: Optional[str] = None, 
        custom_clean_config: Optional[DatasetCleanConfig] = None,
        load_config: Optional[ESLoadConfig] = None
    ) -> PipelineExecutionResult:
        """Execute full end-to-end ETL run."""
        start_time = time.time()
        logs = []

        try:
            # 1. Extract
            logs.append(f"Extracting source: {filename or 'Stream'}")
            df_raw, meta = self.extract(source, filename=filename)
            extracted_rows, extracted_cols = len(df_raw), len(df_raw.columns)
            logs.append(f"Extracted {extracted_rows} rows and {extracted_cols} columns ({meta.get('format', 'unknown')} format)")

            # 2. Profile & Detect
            logs.append("Executing deep semantic type detection and profiling...")
            profile = self.profile_and_detect(df_raw, file_type=meta.get('format', 'csv'))

            # 3. Clean & Transform
            clean_cfg = custom_clean_config or self.generate_default_cleaning_config(profile)
            logs.append("Applying ML-grade transformations and domain standardizers...")
            df_clean, val_report, audit = self.clean_and_transform(df_raw, clean_cfg)
            logs.extend(audit.get('logs', []))

            # 4. Load
            es_cfg = load_config or ESLoadConfig()
            logs.append(f"Ingesting into Elasticsearch index: '{es_cfg.index_name}'...")
            indexed_rows, failed_rows, duration_load, mapping, es_msg = self.load_to_elasticsearch(
                df_clean, clean_cfg, es_cfg
            )
            logs.append(es_msg)

            total_duration = time.time() - start_time
            return PipelineExecutionResult(
                success=True,
                stage="COMPLETED",
                extracted_rows=extracted_rows,
                extracted_cols=extracted_cols,
                cleaned_rows=len(df_clean),
                cleaned_cols=len(df_clean.columns),
                indexed_rows=indexed_rows,
                execution_time_seconds=round(total_duration, 3),
                dataset_profile=profile,
                validation_report=val_report,
                es_mapping=mapping,
                logs=logs
            )

        except Exception as e:
            total_duration = time.time() - start_time
            return PipelineExecutionResult(
                success=False,
                stage="FAILED",
                execution_time_seconds=round(total_duration, 3),
                error_message=str(e),
                logs=logs + [f"Pipeline error: {str(e)}"]
            )
