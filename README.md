# ⚡ Generic ETL & Semantic Type Detection Engine

A domain-agnostic, ML-grade ETL pipeline and interactive Streamlit application that ingests arbitrary data sources (CSV, TSV, Excel, JSON, Parquet), intelligently detects semantic data types via multi-layer fuzzy column name matching and deep pattern probing, applies automated data cleaning/standardization (including Pakistani phone numbers, mixed datetimes, currency, outliers, imputation), and streams cleansed data into Elasticsearch with dynamic schema generation.

---

## Key Features

1. **Multi-Source Universal Extraction (`core/extractor.py`)**:
   - Ingests CSV, TSV, Excel (`.xlsx`, `.xls`), JSON, JSON Lines (`.jsonl`), and Parquet.
   - Automatically sniffs delimiters (`,`, `;`, `\t`, `|`) and character encodings (`utf-8`, `latin-1`, `cp1252`).
   - Recursively flattens semi-structured JSON objects.

2. **Generic Semantic Type Detection Engine (`core/detector/`)**:
   - **Layer 1: Fuzzy Column Matching (`semantic_matcher.py`)**: Uses Levenshtein and Token Sort scoring against a multi-lingual ontology (English, Urdu transliterations, domain acronyms).
   - **Layer 2: Deep Pattern Probing (`pattern_engine.py`)**: Regex evaluation for Pakistani mobile numbers (`03xx`), international phone numbers (`E.164`), CNIC (`XXXXX-XXXXXXX-X`), datetimes, RFC 5322 emails, currency symbols, IP addresses, URLs, and age distributions.
   - **Layer 3: Confidence Score Fusion (`type_inference.py`)**: Computes probabilistic type confidence and generates comprehensive dataset profiles.

3. **ML-Grade Transformation & Data Cleaning (`core/transformer/`)**:
   - **Domain Normalization (`normalizer.py`)**: Standardizes Pakistani mobile numbers (`0300...` / `+92...` -> `923XXXXXXXXX`), mixed datetime formats -> ISO 8601, cleans currency strings -> floats, and normalizes text casing.
   - **Missing Value Imputation (`missing_handler.py`)**: Automated Median (numeric), Mode (categorical), Forward/Backward-Fill (temporal), or constant imputation.
   - **Outlier Handling (`outlier_handler.py`)**: IQR Winsorization/clipping and Z-score capping for numerical features.
   - **Fuzzy Deduplication (`deduplicator.py`)**: Exact and multi-column fuzzy entity resolution.
   - **Quality Validation (`validator.py`)**: Generates pre/post transformation compliance scores and anomaly alerts.

4. **Dynamic Elasticsearch Engine (`core/loader/`)**:
   - Automatically translates detected semantic types into optimized Elasticsearch 8.x/9.x mappings (`text` + `keyword`, `date`, `long`, `double`, `ip`, `boolean`).
   - High-throughput streaming bulk indexer with retry telemetry.
   - **In-Memory Mock ES Engine (`es_client.py`)**: Built-in zero-dependency search engine fallback so full testing and search exploration work out-of-the-box even without a running Elasticsearch daemon.

5. **Interactive Streamlit Dashboard (`app.py`)**:
   - Step 1: Raw Data Extraction & Ingestion (preview, delimiter, encoding, format detection).
   - Step 2: Extracted Data Types & Column Schema (Pandas dtypes, inferred semantic types, null counts, distinct counts).
   - Step 3: Cleaned & Standardized Data (automated normalization, duplicate removal, missing value imputation, CSV/JSON/Excel exports).
   - Step 4: Search & Storage (dynamic Elasticsearch / in-memory search playground).

---

## Quick Start

### 1. Installation & Environment Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Sample Datasets
```bash
python generate_sample_data.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```


---

## 📁 Project Architecture

```
generic_etl_pipeline/
├── venv/
├── README.md
├── requirements.txt
├── app.py                     # Streamlit Main App
├── generate_sample_data.py    # Sample Data Generator
├── core/
│   ├── schemas.py             # Enums, Dataclasses, Profiles
│   ├── extractor.py           # Multi-Format Ingestion Engine
│   ├── detector/
│   │   ├── semantic_matcher.py  # Fuzzy Column Matching
│   │   ├── pattern_engine.py    # Value Regex & Pattern Probing
│   │   └── type_inference.py    # Composite Semantic Inferencer
│   ├── transformer/
│   │   ├── normalizer.py        # PK Phone, Date, CNIC Standardizers
│   │   ├── missing_handler.py   # Smart ML Imputer
│   │   ├── outlier_handler.py   # IQR & Z-score Capping
│   │   ├── deduplicator.py      # Fuzzy Record Linkage
│   │   └── validator.py         # Quality & Schema Conformance
│   ├── loader/
│   │   ├── schema_generator.py  # Dynamic ES Mapping Generator
│   │   ├── es_client.py         # ES Connection & Mock Fallback
│   │   └── bulk_indexer.py      # Streaming Bulk Ingestion
│   └── pipeline.py            # End-to-End ETL Orchestrator
├── ui/
│   └── components.py          # Charts, Metric Cards, Diff Tables
├── data/
│   └── samples/               # Dirty CSV, Excel, JSON Test Datasets
└── tests/
    ├── test_detector.py
    ├── test_transformer.py
    └── test_loader.py
```
