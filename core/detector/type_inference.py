"""
Composite Semantic Type Inference and Dataset Profiler.
Fuses fuzzy column name matching with deep sample value pattern analysis
to accurately determine generic semantic types and data health metrics.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..schemas import ColumnProfile, DatasetProfile, SemanticType
from .semantic_matcher import SemanticColumnMatcher
from .pattern_engine import PatternEngine


class GenericTypeDetector:
    """Intelligent profiler and semantic type inferencer."""

    def __init__(self, sample_size: int = 1000, col_name_weight: float = 0.40, pattern_weight: float = 0.60):
        self.sample_size = sample_size
        self.col_name_weight = col_name_weight
        self.pattern_weight = pattern_weight

    def profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Thoroughly profile and infer the semantic type for a single column."""
        series = df[col]
        total_count = len(series)
        null_count = int(series.isna().sum())
        null_pct = (null_count / total_count) if total_count > 0 else 0.0
        
        # Non-null values
        valid_series = series.dropna()
        unique_count = int(valid_series.nunique())

        # Sample values for deep pattern testing
        sample_size = min(self.sample_size, len(valid_series))
        if sample_size > 0:
            sample_values = valid_series.sample(sample_size, random_state=42).tolist() if len(valid_series) > sample_size else valid_series.tolist()
        else:
            sample_values = []

        # Layer 1: Column name fuzzy matches
        name_matches = SemanticColumnMatcher.match_column_name(col)
        top_name_type, top_name_score = name_matches[0]

        # Layer 2: Value pattern matching
        pattern_ratios = PatternEngine.evaluate_sample_values(sample_values)

        # Basic numeric statistics
        min_val, max_val, mean_val, median_val, std_val = None, None, None, None, None
        outlier_count = 0
        original_dtype = str(series.dtype)

        # Attempt numeric conversion for stats if appropriate
        numeric_series = None
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = valid_series
        else:
            try:
                # Try parsing cleaned floats
                cleaned_numeric = pd.to_numeric(valid_series.astype(str).str.replace(r'[\$,PKRRs\s]', '', regex=True), errors='coerce').dropna()
                if len(cleaned_numeric) > 0.5 * len(valid_series):
                    numeric_series = cleaned_numeric
            except Exception:
                pass

        if numeric_series is not None and len(numeric_series) > 0:
            try:
                min_val = float(numeric_series.min())
                max_val = float(numeric_series.max())
                mean_val = float(numeric_series.mean())
                median_val = float(numeric_series.median())
                std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0

                # IQR Outlier computation
                q25 = numeric_series.quantile(0.25)
                q75 = numeric_series.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outlier_count = int(((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum())
            except Exception:
                pass

        # Layer 3: Decision Fusion Matrix
        detected_type, confidence_score, detected_format, notes = self._fuse_evidence(
            col_name=col,
            original_dtype=original_dtype,
            top_name_type=top_name_type,
            name_score=top_name_score,
            pattern_ratios=pattern_ratios,
            sample_values=sample_values,
            unique_count=unique_count,
            total_count=total_count,
            min_val=min_val,
            max_val=max_val
        )

        # Top frequent values
        top_values = {}
        if len(valid_series) > 0:
            val_counts = valid_series.astype(str).value_counts().head(5).to_dict()
            top_values = {str(k): int(v) for k, v in val_counts.items()}

        return ColumnProfile(
            col_name=col,
            original_dtype=original_dtype,
            detected_type=detected_type,
            confidence_score=round(confidence_score, 3),
            detected_format=detected_format,
            total_count=total_count,
            null_count=null_count,
            null_percentage=round(null_pct, 4),
            unique_count=unique_count,
            sample_values=sample_values[:5],
            top_values=top_values,
            min_value=min_val,
            max_value=max_val,
            mean_value=round(mean_val, 2) if mean_val is not None else None,
            median_value=round(median_val, 2) if median_val is not None else None,
            std_dev=round(std_val, 2) if std_val is not None else None,
            outlier_count=outlier_count,
            regex_match_rate=round(pattern_ratios.get(detected_type, 0.0), 3),
            col_name_similarity=round(top_name_score, 3),
            notes=notes
        )

    def _fuse_evidence(
        self,
        col_name: str,
        original_dtype: str,
        top_name_type: SemanticType,
        name_score: float,
        pattern_ratios: Dict[SemanticType, float],
        sample_values: List[Any],
        unique_count: int,
        total_count: int,
        min_val: Optional[float],
        max_val: Optional[float]
    ) -> Tuple[SemanticType, float, Optional[str], List[str]]:
        """Combine column name similarity and value patterns into final semantic type."""
        notes = []

        # 1. Check Pakistani Phone Match (High Priority)
        pk_phone_ratio = pattern_ratios.get(SemanticType.PHONE_PAKISTAN, 0.0)
        intl_phone_ratio = pattern_ratios.get(SemanticType.PHONE_INTERNATIONAL, 0.0)
        if pk_phone_ratio > 0.40 or (pk_phone_ratio > 0.20 and top_name_type == SemanticType.PHONE_PAKISTAN):
            conf = (0.5 * pk_phone_ratio) + (0.5 * (name_score if top_name_type in [SemanticType.PHONE_PAKISTAN, SemanticType.PHONE_INTERNATIONAL] else 0.4))
            notes.append(f"Pakistani mobile phone pattern identified ({pk_phone_ratio * 100:.1f}% format match)")
            return SemanticType.PHONE_PAKISTAN, min(1.0, conf + 0.2), "03XX-XXXXXXX", notes

        if intl_phone_ratio > 0.60 or (intl_phone_ratio > 0.30 and top_name_type == SemanticType.PHONE_INTERNATIONAL):
            conf = (0.6 * intl_phone_ratio) + (0.4 * name_score)
            notes.append("International telephone format identified")
            return SemanticType.PHONE_INTERNATIONAL, min(1.0, conf), "E.164", notes

        # 2. Check Pakistani CNIC
        cnic_ratio = pattern_ratios.get(SemanticType.CNIC_PAKISTAN, 0.0)
        if cnic_ratio > 0.50 or (cnic_ratio > 0.25 and top_name_type == SemanticType.CNIC_PAKISTAN):
            conf = (0.6 * cnic_ratio) + (0.4 * name_score)
            notes.append("Pakistani CNIC (13-digit identity) pattern detected")
            return SemanticType.CNIC_PAKISTAN, min(1.0, conf + 0.15), "XXXXX-XXXXXXX-X", notes

        # 3. Check Email Address
        email_ratio = pattern_ratios.get(SemanticType.EMAIL, 0.0)
        if email_ratio > 0.60 or (email_ratio > 0.30 and top_name_type == SemanticType.EMAIL):
            conf = (0.7 * email_ratio) + (0.3 * name_score)
            notes.append("RFC 5322 email addresses verified")
            return SemanticType.EMAIL, min(1.0, conf + 0.1), "email@domain.com", notes

        # 4. Check Datetime / Date
        dt_ratio = pattern_ratios.get(SemanticType.DATETIME, 0.0)
        if dt_ratio > 0.60 or (dt_ratio > 0.30 and top_name_type in [SemanticType.DATETIME, SemanticType.DATE]):
            conf = (0.6 * dt_ratio) + (0.4 * name_score)
            target_type = SemanticType.DATE if top_name_type == SemanticType.DATE else SemanticType.DATETIME
            notes.append(f"Parsed datetime format across {dt_ratio * 100:.1f}% sample values")
            return target_type, min(1.0, conf + 0.1), "ISO 8601 / Multi-format", notes

        # 5. Check Age
        age_ratio = pattern_ratios.get(SemanticType.AGE, 0.0)
        if (top_name_type == SemanticType.AGE and name_score > 0.70 and age_ratio > 0.60) or \
           (min_val is not None and max_val is not None and 0 <= min_val and max_val <= 115 and top_name_type == SemanticType.AGE):
            notes.append("Human age distribution (0-115 years) confirmed")
            return SemanticType.AGE, 0.95, "Integer (0-120)", notes

        # 6. Check Currency / Monetary
        curr_ratio = pattern_ratios.get(SemanticType.CURRENCY_AMOUNT, 0.0)
        if curr_ratio > 0.30 or (top_name_type == SemanticType.CURRENCY_AMOUNT and name_score > 0.65):
            conf = (0.5 * curr_ratio) + (0.5 * name_score)
            notes.append("Financial / currency values detected")
            return SemanticType.CURRENCY_AMOUNT, min(1.0, conf + 0.2), "Monetary Amount", notes

        # 7. Check Percentage
        pct_ratio = pattern_ratios.get(SemanticType.PERCENTAGE, 0.0)
        if pct_ratio > 0.40 or (top_name_type == SemanticType.PERCENTAGE and name_score > 0.60):
            return SemanticType.PERCENTAGE, 0.90, "Percentage %", ["Percentage symbols detected"]

        # 8. Check IP Address & URL
        ip_ratio = pattern_ratios.get(SemanticType.IP_ADDRESS, 0.0)
        if ip_ratio > 0.50:
            return SemanticType.IP_ADDRESS, 0.95, "IPv4/IPv6", ["Valid IP network octets"]
        
        url_ratio = pattern_ratios.get(SemanticType.URL, 0.0)
        if url_ratio > 0.50:
            return SemanticType.URL, 0.95, "URL", ["Valid web URLs"]

        # 9. Check Boolean
        bool_ratio = pattern_ratios.get(SemanticType.BOOLEAN, 0.0)
        if bool_ratio > 0.85 and unique_count <= 4:
            return SemanticType.BOOLEAN, 0.95, "Boolean (True/False)", ["Binary state flag values"]

        # 10. Check Address
        addr_ratio = pattern_ratios.get(SemanticType.ADDRESS, 0.0)
        if addr_ratio > 0.35 or (top_name_type == SemanticType.ADDRESS and name_score > 0.70):
            notes.append("Street / postal address markers detected")
            return SemanticType.ADDRESS, 0.85, "Freeform Address", notes

        # 11. Check Categorical (Low Cardinality)
        if (unique_count <= 25 or (total_count > 50 and unique_count / total_count < 0.08)) and original_dtype in ['object', 'category', 'string']:
            notes.append(f"Low cardinality category ({unique_count} distinct categories)")
            return SemanticType.CATEGORICAL, 0.85, "Categorical Enum", notes

        # 12. Check Long Text
        if original_dtype in ['object', 'string'] and sample_values:
            avg_len = sum(len(str(s)) for s in sample_values) / len(sample_values)
            if avg_len > 80:
                return SemanticType.TEXT_LONG, 0.85, "Long Text / Notes", ["Extended narrative / descriptive text"]

        # 13. Numeric Integer / Float fallback
        num_int_ratio = pattern_ratios.get(SemanticType.NUMERIC_INTEGER, 0.0)
        num_float_ratio = pattern_ratios.get(SemanticType.NUMERIC_FLOAT, 0.0)

        if num_int_ratio > 0.85:
            if top_name_type == SemanticType.IDENTIFIER_ID:
                return SemanticType.IDENTIFIER_ID, 0.90, "ID / Key", ["Sequential or discrete identifier"]
            return SemanticType.NUMERIC_INTEGER, 0.85, "Integer", ["Whole numeric values"]

        if num_float_ratio > 0.70 or num_int_ratio + num_float_ratio > 0.80:
            return SemanticType.NUMERIC_FLOAT, 0.85, "Float", ["Continuous numeric values"]

        # 14. Fallback on high-confidence column name match
        if name_score > 0.75:
            return top_name_type, name_score, None, [f"Inferred primarily from column name semantic similarity ({name_score:.2f})"]

        # 15. Default Generic Text / Unknown
        if original_dtype in ['object', 'string']:
            return SemanticType.TEXT_GENERIC, 0.60, "Generic String", ["Standard textual content"]

        return SemanticType.UNKNOWN, 0.40, "Unknown", ["Could not ascertain specific domain type"]

    def profile_dataset(self, df: pd.DataFrame, file_type: str = "csv", file_size_bytes: int = 0) -> DatasetProfile:
        """Run complete profiling and semantic inference across all columns in dataset."""
        columns_profile: Dict[str, ColumnProfile] = {}
        for col in df.columns:
            columns_profile[col] = self.profile_column(df, col)

        duplicate_rows = int(df.duplicated().sum())
        memory_bytes = int(df.memory_usage(deep=True).sum())

        return DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns_profile,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            duplicate_rows_count=duplicate_rows,
            memory_usage_bytes=memory_bytes
        )
