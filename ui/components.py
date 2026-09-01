"""
Reusable UI Components and Visualizations for the Generic ETL Streamlit Dashboard.
Includes statistical metric cards, type distribution charts, data diff viewers,
and interactive search result renderers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional
from core.schemas import ColumnProfile, DatasetProfile, SemanticType, ValidationReport


def render_metric_cards(extracted_rows: int, cleaned_rows: int, cols: int, quality_score: float):
    """Render top-level executive KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw Rows Extracted", f"{extracted_rows:,}")
    with col2:
        st.metric("Cleaned Rows", f"{cleaned_rows:,}", delta=f"{cleaned_rows - extracted_rows}" if cleaned_rows != extracted_rows else "0")
    with col3:
        st.metric("Total Features / Cols", f"{cols}")
    with col4:
        delta_color = "normal" if quality_score >= 80 else "inverse"
        st.metric("Data Quality Score", f"{quality_score:.1f}%", delta=f"{quality_score - 50:.1f}%" if quality_score > 50 else None)


def render_semantic_type_chart(profile: DatasetProfile):
    """Render an interactive Donut chart of inferred semantic data types."""
    type_counts: Dict[str, int] = {}
    for col, p in profile.columns.items():
        type_name = p.detected_type.value.replace('_', ' ').title()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    df_chart = pd.DataFrame(list(type_counts.items()), columns=['Semantic Type', 'Count'])
    
    fig = px.pie(
        df_chart, 
        names='Semantic Type', 
        values='Count',
        hole=0.45,
        title="Detected Semantic Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


def render_missing_values_chart(df_raw: pd.DataFrame, df_clean: Optional[pd.DataFrame] = None):
    """Render Before vs After Missing Value comparison bar chart."""
    raw_nulls = df_raw.isna().sum()
    cols = raw_nulls.index.tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols,
        y=raw_nulls.values,
        name='Raw Missing Count',
        marker_color='#EF553B'
    ))

    if df_clean is not None:
        clean_nulls = df_clean.isna().sum().reindex(cols, fill_value=0)
        fig.add_trace(go.Bar(
            x=cols,
            y=clean_nulls.values,
            name='Cleaned Missing Count',
            marker_color='#00CC96'
        ))

    fig.update_layout(
        barmode='group',
        title="Missing Values (Pre vs Post Transformation)",
        xaxis_title="Column",
        yaxis_title="Null Count",
        margin=dict(t=40, b=20, l=20, r=20),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)


def render_type_badge(sem_type: SemanticType) -> str:
    """Return an HTML styled badge for semantic types."""
    colors = {
        SemanticType.PHONE_PAKISTAN: "#2ECC71",
        SemanticType.PHONE_INTERNATIONAL: "#27AE60",
        SemanticType.DATETIME: "#3498DB",
        SemanticType.DATE: "#2980B9",
        SemanticType.AGE: "#E67E22",
        SemanticType.CURRENCY_AMOUNT: "#F1C40F",
        SemanticType.CNIC_PAKISTAN: "#9B59B6",
        SemanticType.EMAIL: "#1ABC9C",
        SemanticType.CATEGORICAL: "#E74C3C",
        SemanticType.NUMERIC_INTEGER: "#34495E",
        SemanticType.NUMERIC_FLOAT: "#7F8C8D",
        SemanticType.BOOLEAN: "#16A085",
        SemanticType.ADDRESS: "#D35400",
    }
    bg = colors.get(sem_type, "#95A5A6")
    label = sem_type.value.upper().replace('_', ' ')
    return f'<span style="background-color: {bg}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{label}</span>'


def render_column_profile_table(profile: DatasetProfile):
    """Render interactive summary table of all profiled columns."""
    rows = []
    for col, p in profile.columns.items():
        rows.append({
            "Column Name": col,
            "Detected Semantic Type": p.detected_type.value,
            "Confidence": f"{p.confidence_score * 100:.1f}%",
            "Format / Pattern": p.detected_format or "N/A",
            "Null Count (%)": f"{p.null_count} ({p.null_percentage * 100:.1f}%)",
            "Unique Values": p.unique_count,
            "Outliers": p.outlier_count,
            "Key Notes": "; ".join(p.notes[:2]) if p.notes else "Standard"
        })
    df_prof = pd.DataFrame(rows)
    st.dataframe(df_prof, use_container_width=True)
