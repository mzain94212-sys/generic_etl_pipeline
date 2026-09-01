"""
Statistical Outlier Detection and Treatment Engine.
Supports IQR (Interquartile Range) and Z-score methods with clipping/Winsorization,
nullification, or row dropping for numerical features.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..schemas import ColumnCleanConfig, OutlierStrategy, SemanticType


class OutlierHandler:
    """Detects and remedies extreme numerical anomalies."""

    @classmethod
    def handle_outliers(
        cls, 
        df: pd.DataFrame, 
        clean_configs: Dict[str, ColumnCleanConfig]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and treat outliers in numeric columns according to column configs.
        Returns (treated_df, outlier_summary).
        """
        df_out = df.copy()
        summary = {
            'total_outliers_treated': 0,
            'column_stats': {}
        }

        for col in df_out.columns:
            config = clean_configs.get(col)
            if not config or config.outlier_strategy == OutlierStrategy.NONE:
                continue

            # Check if column is numeric
            numeric_s = pd.to_numeric(df_out[col], errors='coerce')
            valid_vals = numeric_s.dropna()
            if len(valid_vals) < 10:
                continue

            strategy = config.outlier_strategy
            k = config.outlier_threshold if config.outlier_threshold > 0 else 1.5
            outlier_count = 0
            lower_bound, upper_bound = None, None

            if strategy in [OutlierStrategy.CLIP_IQR, OutlierStrategy.SET_NULL, OutlierStrategy.DROP]:
                q25 = valid_vals.quantile(0.25)
                q75 = valid_vals.quantile(0.75)
                iqr = q75 - q25
                if iqr <= 0:
                    continue
                lower_bound = q25 - k * iqr
                upper_bound = q75 + k * iqr

            elif strategy == OutlierStrategy.CLIP_ZSCORE:
                mean_v = valid_vals.mean()
                std_v = valid_vals.std()
                if std_v <= 0:
                    continue
                z_thresh = config.outlier_threshold if config.outlier_threshold >= 1.0 else 3.0
                lower_bound = mean_v - z_thresh * std_v
                upper_bound = mean_v + z_thresh * std_v

            if lower_bound is not None and upper_bound is not None:
                mask_outliers = (numeric_s < lower_bound) | (numeric_s > upper_bound)
                outlier_count = int(mask_outliers.sum())

                if outlier_count > 0:
                    if strategy in [OutlierStrategy.CLIP_IQR, OutlierStrategy.CLIP_ZSCORE]:
                        df_out[col] = numeric_s.clip(lower=lower_bound, upper=upper_bound)
                    elif strategy == OutlierStrategy.SET_NULL:
                        df_out.loc[mask_outliers, col] = np.nan
                    elif strategy == OutlierStrategy.DROP:
                        df_out = df_out[~mask_outliers]

                    summary['total_outliers_treated'] += outlier_count
                    summary['column_stats'][col] = {
                        'strategy': strategy.value,
                        'outliers_count': outlier_count,
                        'lower_bound': round(float(lower_bound), 2),
                        'upper_bound': round(float(upper_bound), 2)
                    }

        return df_out, summary
