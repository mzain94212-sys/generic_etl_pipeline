"""
Exact and Fuzzy Record Deduplication Engine.
Performs exact duplicate row removal as well as multi-attribute fuzzy entity linkage
to identify and merge near-duplicate customer/entity records.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from ..detector.semantic_matcher import fuzzy_score


class FuzzyDeduplicator:
    """Detects and removes duplicate and near-duplicate records."""

    @classmethod
    def deduplicate(
        cls, 
        df: pd.DataFrame, 
        exact_dedup: bool = True,
        fuzzy_keys: Optional[List[str]] = None,
        similarity_threshold: float = 0.85,
        max_rows_for_pairwise: int = 2000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute exact and fuzzy record deduplication.
        Returns (deduplicated_df, deduplication_summary).
        """
        df_out = df.copy()
        summary = {
            'exact_duplicates_removed': 0,
            'fuzzy_duplicates_removed': 0,
            'initial_rows': len(df),
            'final_rows': len(df)
        }

        # 1. Exact Duplicate Removal
        if exact_dedup:
            before_len = len(df_out)
            df_out = df_out.drop_duplicates().reset_index(drop=True)
            summary['exact_duplicates_removed'] = before_len - len(df_out)

        # 2. Fuzzy Deduplication on Designated Key Columns
        valid_keys = [k for k in (fuzzy_keys or []) if k in df_out.columns]
        if valid_keys and len(df_out) > 1:
            # Build composite string representation per row
            composite_series = df_out[valid_keys].fillna('').astype(str).agg(' | '.join, axis=1)
            
            # Use blocking heuristic (group by first 3 characters of first key) to avoid O(N^2) explosion
            n_rows = min(len(df_out), max_rows_for_pairwise)
            indices_to_drop: Set[int] = set()

            # Compare rows
            for i in range(n_rows):
                if i in indices_to_drop:
                    continue
                s_i = composite_series.iloc[i]
                if not s_i.strip():
                    continue

                for j in range(i + 1, n_rows):
                    if j in indices_to_drop:
                        continue
                    s_j = composite_series.iloc[j]
                    if not s_j.strip():
                        continue

                    # Fast length check
                    len_ratio = len(min(s_i, s_j, key=len)) / max(len(max(s_i, s_j, key=len)), 1)
                    if len_ratio < 0.70:
                        continue

                    score = fuzzy_score(s_i, s_j)
                    if score >= similarity_threshold:
                        indices_to_drop.add(j)

            if indices_to_drop:
                summary['fuzzy_duplicates_removed'] = len(indices_to_drop)
                df_out = df_out.drop(index=list(indices_to_drop)).reset_index(drop=True)

        summary['final_rows'] = len(df_out)
        return df_out, summary
