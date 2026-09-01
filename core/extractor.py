"""
Generic Data Extractor Module.
Handles multi-format data ingestion: CSV, TSV, Excel (.xlsx, .xls), JSON, JSONL, Parquet, and in-memory streams.
Includes automatic delimiter sniffing, encoding detection, and semi-structured JSON flattening.
"""

import io
import os
import csv
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


class GenericExtractor:
    """Universal data ingestion engine supporting diverse file formats and encodings."""

    SUPPORTED_EXTENSIONS = {
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.txt': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xlsm': 'excel',
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.parquet': 'parquet',
        '.pq': 'parquet',
    }

    ENCODINGS_TO_TRY = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

    @classmethod
    def detect_format(cls, filename: str, content_sample: Optional[bytes] = None) -> str:
        """Infer file format from extension or byte sniffing."""
        ext = os.path.splitext(filename)[1].lower() if filename else ''
        if ext in cls.SUPPORTED_EXTENSIONS:
            return cls.SUPPORTED_EXTENSIONS[ext]

        if content_sample:
            # Check for JSON
            sample_str = content_sample[:500].decode('utf-8', errors='ignore').strip()
            if sample_str.startswith('{') or sample_str.startswith('['):
                return 'json'
            # Check for Parquet magic bytes 'PAR1'
            if content_sample[:4] == b'PAR1':
                return 'parquet'
            # Default to CSV
            return 'csv'

        return 'csv'

    @classmethod
    def sniff_delimiter(cls, sample_text: str) -> str:
        """Automatically detect the CSV delimiter (, ; \t |)."""
        try:
            dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=[',', ';', '\t', '|', ':'])
            return dialect.delimiter
        except Exception:
            # Fallback heuristic: count frequencies
            candidates = [',', ';', '\t', '|']
            counts = {c: sample_text[:2000].count(c) for c in candidates}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else ','

    @classmethod
    def read_csv(
        cls, 
        source: Union[str, io.BytesIO, io.StringIO, bytes], 
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read CSV/TSV data with automatic encoding and delimiter detection."""
        metadata = {'format': 'csv', 'delimiter': ',', 'encoding': 'utf-8'}
        
        # Read raw bytes if necessary
        if isinstance(source, str):
            with open(source, 'rb') as f:
                raw_bytes = f.read()
        elif isinstance(source, io.BytesIO):
            raw_bytes = source.getvalue()
        elif isinstance(source, bytes):
            raw_bytes = source
        elif isinstance(source, io.StringIO):
            raw_bytes = source.getvalue().encode('utf-8')
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Detect encoding
        encodings = [encoding] if encoding else cls.ENCODINGS_TO_TRY
        decoded_text = None
        used_encoding = 'utf-8'

        for enc in encodings:
            if not enc:
                continue
            try:
                decoded_text = raw_bytes.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            decoded_text = raw_bytes.decode('latin-1', errors='replace')
            used_encoding = 'latin-1'

        metadata['encoding'] = used_encoding

        # Sniff delimiter if not specified
        if delimiter is None:
            metadata['delimiter'] = cls.sniff_delimiter(decoded_text)
        else:
            metadata['delimiter'] = delimiter

        # Parse with Pandas
        string_buffer = io.StringIO(decoded_text)
        df = pd.read_csv(
            string_buffer,
            sep=metadata['delimiter'],
            nrows=nrows,
            engine='python',
            on_bad_lines='skip',
            **kwargs
        )
        return df, metadata

    @classmethod
    def read_excel(
        cls, 
        source: Union[str, io.BytesIO, bytes], 
        sheet_name: Optional[Union[str, int]] = 0,
        nrows: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read Excel spreadsheets (.xlsx, .xls) with sheet inspection."""
        metadata = {'format': 'excel', 'sheet_names': []}
        
        # Wrap bytes in BytesIO
        if isinstance(source, bytes):
            source = io.BytesIO(source)

        try:
            excel_file = pd.ExcelFile(source)
            metadata['sheet_names'] = excel_file.sheet_names
            actual_sheet = sheet_name if sheet_name is not None else 0
            df = pd.read_excel(excel_file, sheet_name=actual_sheet, nrows=nrows, **kwargs)
            metadata['selected_sheet'] = str(actual_sheet)
            return df, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file: {str(e)}")

    @classmethod
    def read_json(
        cls, 
        source: Union[str, io.BytesIO, io.StringIO, bytes], 
        flatten_nested: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read JSON / JSON Lines data and optionally flatten nested records."""
        metadata = {'format': 'json', 'flattened': flatten_nested}

        if isinstance(source, str):
            with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif isinstance(source, (io.BytesIO, bytes)):
            content = source.getvalue().decode('utf-8', errors='ignore') if isinstance(source, io.BytesIO) else source.decode('utf-8', errors='ignore')
        elif isinstance(source, io.StringIO):
            content = source.getvalue()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        content_strip = content.strip()
        
        # Check if JSONL (newline delimited)
        if '\n' in content_strip and not (content_strip.startswith('[') and content_strip.endswith(']')):
            lines = [line.strip() for line in content_strip.split('\n') if line.strip()]
            records = []
            for line in lines:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if flatten_nested:
                df = pd.json_normalize(records)
            else:
                df = pd.DataFrame(records)
            metadata['format'] = 'jsonl'
            return df, metadata

        # Standard JSON array or object
        parsed_data = json.loads(content_strip)
        if isinstance(parsed_data, dict):
            # If wrapped in a data/records key
            for key in ['data', 'records', 'items', 'results', 'rows']:
                if key in parsed_data and isinstance(parsed_data[key], list):
                    parsed_data = parsed_data[key]
                    break
            else:
                parsed_data = [parsed_data]

        if flatten_nested:
            df = pd.json_normalize(parsed_data)
        else:
            df = pd.DataFrame(parsed_data)

        return df, metadata

    @classmethod
    def read_parquet(
        cls, 
        source: Union[str, io.BytesIO, bytes], 
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read Apache Parquet columnar files."""
        metadata = {'format': 'parquet'}
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        df = pd.read_parquet(source, **kwargs)
        return df, metadata

    @classmethod
    def extract(
        cls, 
        source: Union[str, io.BytesIO, io.StringIO, bytes], 
        filename: Optional[str] = None,
        format_hint: Optional[str] = None,
        sheet_name: Optional[Union[str, int]] = 0,
        delimiter: Optional[str] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Unified extraction entry point.
        Automatically detects format, parses the data, cleans header names, and returns (DataFrame, metadata).
        """
        # Determine format
        if format_hint:
            fmt = format_hint.lower()
        elif filename:
            fmt = cls.detect_format(filename)
        else:
            fmt = 'csv'

        if fmt in ['csv', 'tsv']:
            df, meta = cls.read_csv(source, delimiter=delimiter, **kwargs)
        elif fmt in ['excel', 'xlsx', 'xls']:
            df, meta = cls.read_excel(source, sheet_name=sheet_name, **kwargs)
        elif fmt in ['json', 'jsonl']:
            df, meta = cls.read_json(source, **kwargs)
        elif fmt in ['parquet', 'pq']:
            df, meta = cls.read_parquet(source, **kwargs)
        else:
            # Fallback to CSV
            df, meta = cls.read_csv(source, delimiter=delimiter, **kwargs)

        # Standardize DataFrame column names: strip whitespace
        df.columns = [str(c).strip() for c in df.columns]
        meta['row_count'] = len(df)
        meta['col_count'] = len(df.columns)
        return df, meta
