"""
Data Quality and Validation Engine.
Performs pre/post-transformation integrity checks, schema conformance scoring,
and generates comprehensive data quality reports.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from ..schemas import ColumnCleanConfig, SemanticType, ValidationReport
from ..detector.pattern_engine import PatternEngine


class DataQualityValidator:
    """Evaluates transformation quality, compliance, and anomaly rates."""

    @classmethod
    def validate_dataset(
        cls, 
        df: pd.DataFrame, 
        clean_configs: Dict[str, ColumnCleanConfig]
    ) -> ValidationReport:
        """Run all data quality and domain rule validations on the DataFrame."""
        column_reports: Dict[str, Dict[str, Any]] = {}
        anomalies: List[str] = []
        total_checks = 0
        passed_checks = 0

        for col in df.columns:
            config = clean_configs.get(col)
            sem_type = config.target_semantic_type if config else SemanticType.UNKNOWN
            series = df[col]
            valid_vals = series.dropna()

            total_col_rows = len(series)
            null_count = int(series.isna().sum())
            col_valid_count = 0
            col_invalid_count = 0
            check_name = "General Type Check"

            if len(valid_vals) == 0:
                column_reports[col] = {
                    'semantic_type': sem_type.value,
                    'validation_type': 'all_null',
                    'compliance_rate': 0.0,
                    'valid_count': 0,
                    'invalid_count': total_col_rows
                }
                total_checks += 1
                continue

            # Semantic Type specific validations
            if sem_type == SemanticType.PHONE_PAKISTAN:
                check_name = "PK Phone Format (923XXXXXXXXX / +923XXXXXXXXX)"
                for v in valid_vals:
                    sv = str(v)
                    if (sv.startswith('923') and len(sv) == 12) or (sv.startswith('+923') and len(sv) == 13) or (sv.startswith('03') and len(sv) == 11):
                        col_valid_count += 1
                    else:
                        col_invalid_count += 1

            elif sem_type == SemanticType.EMAIL:
                check_name = "RFC 5322 Email Format"
                for v in valid_vals:
                    if PatternEngine.test_email(str(v)):
                        col_valid_count += 1
                    else:
                        col_invalid_count += 1

            elif sem_type == SemanticType.CNIC_PAKISTAN:
                check_name = "Pakistani CNIC (XXXXX-XXXXXXX-X)"
                for v in valid_vals:
                    sv = str(v)
                    if len(sv) == 15 and sv[5] == '-' and sv[13] == '-':
                        col_valid_count += 1
                    else:
                        col_invalid_count += 1

            elif sem_type in [SemanticType.DATETIME, SemanticType.DATE]:
                check_name = "Valid ISO/Target Datetime"
                for v in valid_vals:
                    is_dt, _ = PatternEngine.test_datetime(str(v))
                    if is_dt:
                        col_valid_count += 1
                    else:
                        col_invalid_count += 1

            elif sem_type == SemanticType.AGE:
                check_name = "Age Range (0 - 120)"
                for v in valid_vals:
                    try:
                        val_num = float(v)
                        if 0 <= val_num <= 120:
                            col_valid_count += 1
                        else:
                            col_invalid_count += 1
                    except ValueError:
                        col_invalid_count += 1

            elif sem_type in [SemanticType.NUMERIC_FLOAT, SemanticType.CURRENCY_AMOUNT]:
                check_name = "Numeric Float Casting"
                for v in valid_vals:
                    try:
                        float(v)
                        col_valid_count += 1
                    except ValueError:
                        col_invalid_count += 1

            else:
                # Default non-empty string check
                col_valid_count = len(valid_vals)
                col_invalid_count = 0

            compliance = (col_valid_count / len(valid_vals)) if len(valid_vals) > 0 else 1.0
            total_checks += len(valid_vals)
            passed_checks += col_valid_count

            if col_invalid_count > 0:
                anomalies.append(f"Column '{col}' has {col_invalid_count} records failing {check_name}")

            column_reports[col] = {
                'semantic_type': sem_type.value,
                'validation_type': check_name,
                'compliance_rate': round(compliance * 100, 2),
                'valid_count': col_valid_count,
                'invalid_count': col_invalid_count,
                'null_count': null_count
            }

        overall_passed = (passed_checks / total_checks >= 0.80) if total_checks > 0 else True

        return ValidationReport(
            passed=overall_passed,
            total_validations=total_checks,
            valid_count=passed_checks,
            invalid_count=total_checks - passed_checks,
            column_reports=column_reports,
            anomalies=anomalies
        )
