"""
ML-Grade Missing Value Imputation and Treatment Engine.
Provides automated and strategy-driven imputation for categorical, numeric,
temporal, and text variables, with outlier-robust statistics.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..schemas import ColumnCleanConfig, DatasetCleanConfig, ImputationStrategy, SemanticType


class MissingValueHandler:
    """Handles missing value detection, threshold filtering, and intelligent imputation."""

    @classmethod
    def apply_imputation(
        cls, 
        df: pd.DataFrame, 
        clean_configs: Dict[str, ColumnCleanConfig],
        global_config: Optional[DatasetCleanConfig] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing values across all columns according to column configs and semantic types.
        Returns (imputed_df, imputation_summary).
        """
        df_out = df.copy()
        summary = {
            'imputed_cells': 0,
            'dropped_rows': 0,
            'dropped_columns': [],
            'column_stats': {}
        }

        # 1. High-Null Column Filtering
        if global_config and global_config.drop_columns_with_high_nulls:
            null_ratio = df_out.isna().mean()
            cols_to_drop = null_ratio[null_ratio > global_config.null_column_drop_threshold].index.tolist()
            if cols_to_drop:
                df_out = df_out.drop(columns=cols_to_drop)
                summary['dropped_columns'] = cols_to_drop

        # 2. High-Null Row Filtering
        if global_config and global_config.drop_rows_with_high_nulls:
            row_null_ratio = df_out.isna().mean(axis=1)
            initial_rows = len(df_out)
            df_out = df_out[row_null_ratio <= global_config.null_row_drop_threshold]
            summary['dropped_rows'] += (initial_rows - len(df_out))

        # 3. Column-by-Column Imputation
        for col in df_out.columns:
            config = clean_configs.get(col)
            null_count = int(df_out[col].isna().sum())
            if null_count == 0:
                summary['column_stats'][col] = {'strategy': 'none', 'nulls_handled': 0}
                continue

            strategy = config.imputation_strategy if config else ImputationStrategy.AUTO
            sem_type = config.target_semantic_type if config else SemanticType.UNKNOWN

            imputed_value = None
            used_strategy = strategy

            # Resolve AUTO strategy based on semantic type
            if strategy == ImputationStrategy.AUTO:
                if sem_type in [SemanticType.NUMERIC_FLOAT, SemanticType.CURRENCY_AMOUNT, SemanticType.PERCENTAGE]:
                    used_strategy = ImputationStrategy.MEDIAN
                elif sem_type in [SemanticType.NUMERIC_INTEGER, SemanticType.AGE]:
                    used_strategy = ImputationStrategy.MEDIAN
                elif sem_type in [SemanticType.CATEGORICAL, SemanticType.GENDER, SemanticType.BOOLEAN, SemanticType.CITY, SemanticType.COUNTRY]:
                    used_strategy = ImputationStrategy.MODE
                elif sem_type in [SemanticType.DATETIME, SemanticType.DATE]:
                    used_strategy = ImputationStrategy.FORWARD_FILL
                else:
                    used_strategy = ImputationStrategy.CONSTANT
                    imputed_value = "Unknown"

            # Execute the strategy
            if used_strategy == ImputationStrategy.MEAN:
                numeric_s = pd.to_numeric(df_out[col], errors='coerce')
                mean_v = numeric_s.mean()
                if not pd.isna(mean_v):
                    df_out[col] = df_out[col].fillna(round(mean_v, 2))
                    imputed_value = round(mean_v, 2)

            elif used_strategy == ImputationStrategy.MEDIAN:
                numeric_s = pd.to_numeric(df_out[col], errors='coerce')
                median_v = numeric_s.median()
                if not pd.isna(median_v):
                    if sem_type in [SemanticType.NUMERIC_INTEGER, SemanticType.AGE]:
                        median_v = int(round(median_v))
                    else:
                        median_v = round(median_v, 2)
                    df_out[col] = df_out[col].fillna(median_v)
                    imputed_value = median_v

            elif used_strategy == ImputationStrategy.MODE:
                modes = df_out[col].mode()
                if len(modes) > 0:
                    mode_v = modes[0]
                    df_out[col] = df_out[col].fillna(mode_v)
                    imputed_value = str(mode_v)
                else:
                    df_out[col] = df_out[col].fillna("Unknown")
                    imputed_value = "Unknown"

            elif used_strategy == ImputationStrategy.FORWARD_FILL:
                df_out[col] = df_out[col].ffill().bfill()
                imputed_value = "forward_fill"

            elif used_strategy == ImputationStrategy.BACKWARD_FILL:
                df_out[col] = df_out[col].bfill().ffill()
                imputed_value = "backward_fill"

            elif used_strategy == ImputationStrategy.CONSTANT:
                const_v = config.imputation_constant_value if config and config.imputation_constant_value is not None else "Unknown"
                df_out[col] = df_out[col].fillna(const_v)
                imputed_value = const_v

            elif used_strategy == ImputationStrategy.DROP_ROW:
                initial_len = len(df_out)
                df_out = df_out.dropna(subset=[col])
                summary['dropped_rows'] += (initial_len - len(df_out))
                imputed_value = "dropped_rows"

            # Post-check handled count
            remaining_nulls = int(df_out[col].isna().sum())
            handled = null_count - remaining_nulls
            summary['imputed_cells'] += max(0, handled)
            summary['column_stats'][col] = {
                'strategy': used_strategy.value if hasattr(used_strategy, 'value') else str(used_strategy),
                'nulls_handled': handled,
                'imputed_value': imputed_value
            }

        return df_out, summary
