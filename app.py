"""
Generic ETL Pipeline - Raw Data Extraction, Type Detection & Cleaning.
Universal multi-format data extraction, data type inspection, and automated data cleaning.
"""

import io
import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

from core.schemas import (
    ColumnCleanConfig,
    DatasetCleanConfig,
    ESLoadConfig,
    ImputationStrategy,
    OutlierStrategy,
    SemanticType,
)
from core.pipeline import GenericETLPipeline
from core.loader.schema_generator import DynamicESMappingGenerator
from core.loader.es_client import ESClientManager, MockElasticsearchClient
from ui.components import (
    render_metric_cards,
    render_type_badge
)

# Set page config
st.set_page_config(
    page_title="Generic ETL Pipeline | Extraction & Cleaning",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1F2937;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = GenericETLPipeline()
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'meta' not in st.session_state:
    st.session_state.meta = {}
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'clean_config' not in st.session_state:
    st.session_state.clean_config = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = []
if 'es_loaded' not in st.session_state:
    st.session_state.es_loaded = False
if 'es_client_instance' not in st.session_state:
    st.session_state.es_client_instance = None


def extract_and_process_data(df: pd.DataFrame, meta: dict, auto_clean: bool = True):
    """Profile data types and perform automated cleaning."""
    st.session_state.raw_df = df
    st.session_state.meta = meta
    st.session_state.profile = st.session_state.pipeline.profile_and_detect(
        df, file_type=meta.get('format', 'csv')
    )
    st.session_state.clean_config = st.session_state.pipeline.generate_default_cleaning_config(
        st.session_state.profile
    )
    
    if auto_clean:
        cleaned_df, _, audit = st.session_state.pipeline.clean_and_transform(
            df, st.session_state.clean_config
        )
        st.session_state.cleaned_df = cleaned_df
        st.session_state.audit_logs = audit.get('logs', [])
        st.session_state.audit_summary = audit
    else:
        st.session_state.cleaned_df = None
        st.session_state.audit_logs = []
    
    st.session_state.es_loaded = False


def main():
    st.markdown('<div class="main-title">⚡ Generic ETL Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Raw Data Extraction, Data Type Extraction & Automated Cleaning</div>', unsafe_allow_html=True)

    # ------------------ SIDEBAR CONFIGURATION ------------------
    with st.sidebar:
        st.header("📂 Data Source Selection")
        source_mode = st.radio(
            "Choose Source Type",
            ["📁 Upload Custom File", "🧪 Load Built-in Samples"],
            index=1
        )

        uploaded_file = None
        sample_choice = None

        if source_mode == "📁 Upload Custom File":
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=['csv', 'tsv', 'xlsx', 'xls', 'json', 'jsonl', 'parquet'],
                help="Supports CSV, TSV, Excel (.xlsx, .xls), JSON, JSONL, and Parquet"
            )
        else:
            sample_choice = st.selectbox(
                "Select Sample Dataset",
                [
                    "🇵🇰 Pakistan Customer CRM (03xx Phones, Mixed Dates, CNICs, Salaries)",
                    "⏱️ Call Center Logs (Duration, 12-hr AM/PM times, PKR Costs)",
                    "📊 Sales & Inventory (Excel format with outliers & prices)",
                    "📡 IoT Telemetry (JSON with IPs & Temperatures)"
                ]
            )

        st.divider()
        st.header("⚙️ Search & Storage (Optional)")
        es_host = st.text_input("ES Host URL", value="http://localhost:9200")
        es_index = st.text_input("Target Index Name", value="pakistan_customers_clean")
        use_mock = st.checkbox("Enable In-Memory Search Fallback", value=True)

        st.session_state.es_load_config = ESLoadConfig(
            host=es_host,
            index_name=es_index,
            use_mock_if_unavailable=use_mock,
            overwrite_index=True
        )

    # ------------------ EXTRACTION TRIGGER LOGIC ------------------
    if source_mode == "📁 Upload Custom File" and uploaded_file is not None:
        if st.sidebar.button("📥 Extract Raw Data", type="primary", use_container_width=True):
            with st.spinner("Extracting raw dataset and inspecting types..."):
                raw_bytes = uploaded_file.getvalue()
                df, meta = st.session_state.pipeline.extract(raw_bytes, filename=uploaded_file.name)
                extract_and_process_data(df, meta, auto_clean=True)
                st.success(f"Extracted {len(df):,} rows and {len(df.columns)} columns successfully!")

    elif source_mode == "🧪 Load Built-in Samples" and sample_choice is not None:
        if st.sidebar.button("🚀 Load & Extract Dataset", type="primary", use_container_width=True) or st.session_state.raw_df is None:
            with st.spinner("Loading sample dataset and extracting schema..."):
                samples_dir = "/home/muhammad-zain/generic_etl_pipeline/data/samples"
                os.makedirs(samples_dir, exist_ok=True)
                
                if "Pakistan Customer" in sample_choice:
                    filepath = os.path.join(samples_dir, "pakistan_customers_dirty.csv")
                    st.session_state.es_load_config.index_name = "pakistan_customers_clean"
                elif "Call Center" in sample_choice:
                    filepath = os.path.join(samples_dir, "test.csv")
                    st.session_state.es_load_config.index_name = "call_center_clean"
                elif "Sales" in sample_choice:
                    filepath = os.path.join(samples_dir, "sales_inventory_dirty.xlsx")
                    st.session_state.es_load_config.index_name = "sales_inventory_clean"
                else:
                    filepath = os.path.join(samples_dir, "iot_telemetry_corrupted.json")
                    st.session_state.es_load_config.index_name = "iot_telemetry_clean"

                if os.path.exists(filepath):
                    df, meta = st.session_state.pipeline.extract(filepath, filename=os.path.basename(filepath))
                else:
                    df = pd.DataFrame({
                        "cust_name": ["Muhammad Zain", "Ali Hassan", "Fatima Noor", "Usman Tariq"],
                        "mob_no": ["03001234567", "+92 321 9876543", "00923335551234", "3451122334"],
                        "nic_no": ["35201-1234567-1", "3520298765431", "42101-5544332-9", None],
                        "user_age": [28, 34, 22, 145],
                        "salary": ["PKR 145,000", "Rs. 95000", "$1,200", "120,000"],
                        "joined_date": ["2024-01-15", "15/08/2023", "2023/11/20 14:30:00", None]
                    })
                    meta = {'format': 'csv', 'delimiter': ',', 'encoding': 'utf-8'}

                extract_and_process_data(df, meta, auto_clean=True)

    # Check if raw data exists
    if st.session_state.raw_df is None:
        st.info("👈 Please select a dataset from the sidebar to extract and clean data.")
        return

    raw_df = st.session_state.raw_df
    profile = st.session_state.profile
    cleaned_df = st.session_state.cleaned_df

    # ------------------ UNIFIED VIEW / TABS ------------------
    tab_raw, tab_types, tab_clean, tab_search = st.tabs([
        "📥 1. Raw Data Extraction",
        "🔍 2. Extracted Data Types",
        "✨ 3. Cleaned Data",
        "🚀 4. Search & Export"
    ])

    # ==================== 1. RAW DATA EXTRACTION ====================
    with tab_raw:
        st.markdown('<div class="section-header">📥 Stage 1: Raw Data Extraction & Ingestion</div>', unsafe_allow_html=True)
        
        # Metadata KPI cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Detected File Format", str(st.session_state.meta.get('format', 'CSV')).upper())
        with c2:
            st.metric("Delimiter / Enclosing", str(st.session_state.meta.get('delimiter', 'N/A')))
        with c3:
            st.metric("Character Encoding", str(st.session_state.meta.get('encoding', 'utf-8')))
        with c4:
            st.metric("Raw Shape", f"{len(raw_df):,} rows × {len(raw_df.columns)} cols")

        st.subheader("Raw Ingested Dataset Preview")
        st.dataframe(raw_df, use_container_width=True)

        with st.expander("ℹ️ Raw Ingestion Metadata Details"):
            st.json({
                "format": st.session_state.meta.get('format', 'csv'),
                "delimiter": st.session_state.meta.get('delimiter', ','),
                "encoding": st.session_state.meta.get('encoding', 'utf-8'),
                "total_rows": len(raw_df),
                "total_columns": len(raw_df.columns),
                "column_names": list(raw_df.columns),
                "memory_usage_kb": round(raw_df.memory_usage(deep=True).sum() / 1024, 2)
            })

    # ==================== 2. EXTRACTED DATA TYPES ====================
    with tab_types:
        st.markdown('<div class="section-header">🔍 Stage 2: Extracted Column Data Types</div>', unsafe_allow_html=True)
        st.write("Summary of extracted schema, inferred data types, null statistics, and distinct value counts:")

        type_rows = []
        for col in raw_df.columns:
            col_prof = profile.columns.get(col) if profile else None
            pandas_dtype = str(raw_df[col].dtype)
            detected_type = col_prof.detected_type.value if col_prof else "generic"
            null_count = int(raw_df[col].isna().sum())
            null_pct = f"{(null_count / len(raw_df)) * 100:.1f}%"
            unique_count = int(raw_df[col].nunique(dropna=True))
            
            # Non-null sample
            non_nulls = raw_df[col].dropna()
            sample_val = str(non_nulls.iloc[0]) if len(non_nulls) > 0 else "N/A"

            type_rows.append({
                "Column Name": col,
                "Pandas Dtype": pandas_dtype,
                "Extracted Data Type": detected_type,
                "Null Count": null_count,
                "Null %": null_pct,
                "Unique Values": unique_count,
                "Sample Value": sample_val
            })

        df_types = pd.DataFrame(type_rows)
        st.dataframe(df_types, use_container_width=True)

    # ==================== 3. CLEANED DATA ====================
    with tab_clean:
        st.markdown('<div class="section-header">✨ Stage 3: Cleaned & Standardized Data</div>', unsafe_allow_html=True)

        if cleaned_df is None:
            if st.button("🚀 Run Data Cleaning", type="primary"):
                with st.spinner("Applying automated cleaning and standardization..."):
                    cleaned_df, _, audit = st.session_state.pipeline.clean_and_transform(
                        raw_df, st.session_state.clean_config
                    )
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.audit_logs = audit.get('logs', [])
                    st.rerun()
        else:
            # Summary Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Raw Rows", f"{len(raw_df):,}")
            with c2:
                st.metric("Cleaned Rows", f"{len(cleaned_df):,}")
            with c3:
                dups_removed = len(raw_df) - len(cleaned_df)
                st.metric("Duplicate Rows Removed", f"{dups_removed}")
            with c4:
                st.metric("Cleaned Columns", f"{len(cleaned_df.columns)}")

            # Applied Cleaning Logs
            if st.session_state.audit_logs:
                with st.expander("📋 Applied Cleaning Steps & Rules Summary", expanded=False):
                    for log_msg in st.session_state.audit_logs:
                        st.write(f"✓ {log_msg}")

            st.subheader("Cleaned Dataset Preview")
            st.dataframe(cleaned_df, use_container_width=True)

            # Export Buttons
            st.divider()
            st.subheader("📥 Export Cleaned Dataset")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    data=csv_bytes,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_col2:
                json_bytes = cleaned_df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download as JSON",
                    data=json_bytes,
                    file_name="cleaned_dataset.json",
                    mime="application/json",
                    use_container_width=True
                )
            with exp_col3:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    cleaned_df.to_excel(writer, index=False)
                st.download_button(
                    "📥 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name="cleaned_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    # ==================== 4. SEARCH & ELASTICSEARCH ====================
    with tab_search:
        st.markdown('<div class="section-header">🚀 Stage 4: Search & Indexing</div>', unsafe_allow_html=True)
        
        if cleaned_df is None:
            st.warning("⚠️ Please extract and clean data first.")
        else:
            es_cfg = st.session_state.es_load_config

            btn_col1, _ = st.columns([1, 2])
            with btn_col1:
                if st.button("🚀 Index Cleaned Data into Search Engine", type="primary", use_container_width=True):
                    with st.spinner("Indexing documents..."):
                        success_n, failed_n, duration, mapping, status_msg = st.session_state.pipeline.load_to_elasticsearch(
                            cleaned_df,
                            st.session_state.clean_config,
                            es_cfg
                        )
                        st.session_state.es_loaded = True
                        st.session_state.es_status_msg = status_msg
                        st.session_state.es_client_instance, _, _ = ESClientManager.get_client(es_cfg)
                        st.success(f"✓ {status_msg}")

            if st.session_state.get('es_loaded', False):
                st.divider()
                st.subheader("🔍 Interactive Search Explorer")
                search_term = st.text_input("Enter Search Term (Name, Phone, CNIC, City, etc.)", value="")
                
                client = st.session_state.es_client_instance
                if client:
                    query_body = {
                        "query": {
                            "multi_match": {
                                "query": search_term,
                                "fields": ["*"]
                            }
                        }
                    } if search_term.strip() else {"query": {"match_all": {}}}
                    
                    search_res = client.search(
                        index=es_cfg.index_name.lower().replace(' ', '_'),
                        query=query_body,
                        size=50
                    )
                    total_hits = search_res.get('hits', {}).get('total', {}).get('value', 0)
                    hits = search_res.get('hits', {}).get('hits', [])

                    st.info(f"Found **{total_hits}** matching records in index `{es_cfg.index_name}`")

                    if hits:
                        hit_docs = [h.get('_source', {}) for h in hits]
                        df_hits = pd.DataFrame(hit_docs)
                        st.dataframe(df_hits, use_container_width=True)


if __name__ == "__main__":
    main()
