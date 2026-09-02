"""
Generic Data Normalizer and Domain Standardizer.
Contains domain-specific standardizers for Pakistani and International phone numbers,
datetimes, Pakistani CNICs, currencies, emails, booleans, and text strings.
"""

import re
import datetime
from typing import Any, Optional, Union
import pandas as pd
import dateutil.parser

# Optional phonenumbers import
try:
    import phonenumbers
    HAS_PHONENUMBERS = True
except ImportError:
    HAS_PHONENUMBERS = False


class GenericNormalizer:
    """Standardizes heterogeneous data values into canonical formats."""

    @classmethod
    def clean_pakistan_phone(
        cls, 
        val: Any, 
        prefix: str = "92", 
        null_placeholder: Optional[str] = None
    ) -> Optional[str]:
        """
        Standardize Pakistani phone numbers:
        Handles formats like '03001234567', '0300-1234567', '+92 300 1234567',
        '00923001234567', '3001234567', '92-300-1234567'.
        Returns formatted canonical string e.g. '923001234567' or '+923001234567'.
        Handles nulls and unparseable values gracefully.
        """
        if val is None or pd.isna(val):
            return null_placeholder
        
        s = str(val).strip()
        if not s or s.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return null_placeholder

        # Extract only digits
        digits = re.sub(r'\D', '', s)
        if not digits:
            return null_placeholder

        # Remove leading zeroes
        digits = digits.lstrip('0')

        # If starts with 92 and has 12 digits (92 + 3XX + XXXXXXX)
        if digits.startswith('92') and len(digits) == 12 and digits[2] == '3':
            body = digits[2:]  # 3XXXXXXXXX
        elif len(digits) == 10 and digits.startswith('3'):
            # Form: 3001234567
            body = digits
        elif len(digits) == 11 and digits.startswith('03'):
            # Form: 03001234567
            body = digits[1:]
        elif len(digits) >= 10 and '3' in digits[:3]:
            idx = digits.find('3')
            if idx != -1 and len(digits[idx:]) == 10:
                body = digits[idx:]
            else:
                body = digits
        else:
            body = digits

        # Format with prefix
        p = prefix.strip()
        if p == "+92":
            return f"+92{body}"
        elif p == "92":
            return f"92{body}"
        elif p == "03":
            return f"0{body}"
        elif p == "formatted":
            if len(body) == 10:
                return f"+92 {body[:3]} {body[3:]}"
            return f"+92 {body}"
        else:
            return f"{p}{body}"

    @classmethod
    def clean_international_phone(
        cls, 
        val: Any, 
        default_region: str = "PK", 
        null_placeholder: Optional[str] = None
    ) -> Optional[str]:
        """Parse and standardize international phone numbers to E.164 (+1234567890)."""
        if val is None or pd.isna(val):
            return null_placeholder
        s = str(val).strip()
        if not s or s.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return null_placeholder

        if HAS_PHONENUMBERS:
            try:
                parsed = phonenumbers.parse(s, default_region)
                if phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            except Exception:
                pass

        # Fallback regex standardizer
        digits = re.sub(r'[^\d+]', '', s)
        return digits if len(digits) >= 7 else null_placeholder

    @classmethod
    def clean_pakistan_cnic(cls, val: Any, null_placeholder: Optional[str] = None) -> Optional[str]:
        """Standardize Pakistani 13-digit CNIC into 'XXXXX-XXXXXXX-X'."""
        if val is None or pd.isna(val):
            return null_placeholder
        s = str(val).strip()
        digits = re.sub(r'\D', '', s)
        if len(digits) == 13:
            return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
        return s if s else null_placeholder

    @classmethod
    def clean_datetime(
        cls, 
        val: Any, 
        target_format: str = "%Y-%m-%d %H:%M:%S", 
        null_placeholder: Optional[str] = None
    ) -> Optional[str]:
        """Parse mixed date strings into uniform datetime format."""
        if val is None or pd.isna(val):
            return null_placeholder
        
        # If already a pandas or python timestamp
        if isinstance(val, (pd.Timestamp, pd.DatetimeIndex)):
            return val.strftime(target_format)

        s = str(val).strip()
        if not s or s.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return null_placeholder

        # Check if epoch number
        if s.isdigit():
            num = int(s)
            if len(s) == 10:
                dt = pd.to_datetime(num, unit='s', errors='coerce')
                return dt.strftime(target_format) if not pd.isna(dt) else null_placeholder
            elif len(s) == 13:
                dt = pd.to_datetime(num, unit='ms', errors='coerce')
                return dt.strftime(target_format) if not pd.isna(dt) else null_placeholder

        try:
            parsed = dateutil.parser.parse(s, fuzzy=False)
            return parsed.strftime(target_format)
        except Exception:
            try:
                dt = pd.to_datetime(s, errors='coerce')
                return dt.strftime(target_format) if not pd.isna(dt) else null_placeholder
            except Exception:
                return null_placeholder

    @classmethod
    def clean_duration(
        cls, 
        val: Any, 
        target_format: str = "H:MM:SS", 
        null_placeholder: Optional[str] = None
    ) -> Optional[str]:
        """
        Standardize time and duration values.
        Converts 12-hour AM/PM formats (e.g. '12:36:53 AM' -> '0:36:53', '12:36:53 PM' -> '12:36:53',
        '01:15:30 AM' -> '1:15:30', '01:15:30 PM' -> '13:15:30', '12:00:00 AM' -> '0:00:00') into standard duration format.
        Also handles numeric seconds, timedeltas, natural language ('1h 30m'), and standard 'H:MM:SS' strings.
        Supports target_format: 'H:MM:SS' (e.g. '0:36:53') or 'HH:MM:SS' (e.g. '00:36:53').
        """
        if val is None or pd.isna(val):
            return null_placeholder
        
        # If already a pandas or python timedelta / time
        if isinstance(val, (pd.Timedelta, pd.TimedeltaIndex)):
            total_sec = int(val.total_seconds())
            hours = total_sec // 3600
            minutes = (total_sec % 3600) // 60
            seconds = total_sec % 60
            if target_format == "HH:MM:SS":
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        if isinstance(val, (datetime.time, datetime.datetime)):
            h = val.hour
            m = val.minute
            s_val = val.second
            if target_format == "HH:MM:SS":
                return f"{h:02d}:{m:02d}:{s_val:02d}"
            return f"{h}:{m:02d}:{s_val:02d}"

        s = str(val).strip()
        if not s or s.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return null_placeholder

        # Check numeric seconds (integer or float string)
        if s.isdigit():
            total_sec = int(s)
            hours = total_sec // 3600
            minutes = (total_sec % 3600) // 60
            seconds = total_sec % 60
            if target_format == "HH:MM:SS":
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        # 1. 12-hour format with AM/PM (e.g., '12:36:53 AM', '12:36:53 PM', '1:30:00 PM', '12:36 AM')
        am_pm_match = re.match(r'^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\s*([ap]m)$', s, re.IGNORECASE)
        if am_pm_match:
            h = int(am_pm_match.group(1))
            m = int(am_pm_match.group(2))
            sec = int(am_pm_match.group(3)) if am_pm_match.group(3) is not None else 0
            ampm = am_pm_match.group(4).lower()
            if ampm == 'am':
                if h == 12:
                    h = 0
            elif ampm == 'pm':
                if h != 12:
                    h += 12
            if target_format == "HH:MM:SS":
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{h}:{m:02d}:{sec:02d}"

        # 2. Match standard H:MM:SS or HH:MM:SS or H:MM (24-hour format)
        match_24 = re.match(r'^(\d+):(\d{1,2})(?::(\d{1,2}))?$', s)
        if match_24:
            h = int(match_24.group(1))
            m = int(match_24.group(2))
            sec = int(match_24.group(3)) if match_24.group(3) is not None else 0
            if target_format == "HH:MM:SS":
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{h}:{m:02d}:{sec:02d}"

        # 3. Match natural language duration like '1h 30m 15s', '36m 53s', '45s'
        nat_match = re.findall(r'(\d+)\s*(h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)', s, re.IGNORECASE)
        if nat_match:
            h, m, sec = 0, 0, 0
            for num, unit in nat_match:
                u = unit.lower()
                n = int(num)
                if 'h' in u:
                    h += n
                elif 'm' in u:
                    m += n
                elif 's' in u:
                    sec += n
            m += sec // 60
            sec = sec % 60
            h += m // 60
            m = m % 60
            if target_format == "HH:MM:SS":
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{h}:{m:02d}:{sec:02d}"

        # 4. Fallback using dateutil or date parsing if only time
        try:
            parsed = dateutil.parser.parse(s, fuzzy=False)
            h = parsed.hour
            m = parsed.minute
            sec = parsed.second
            if target_format == "HH:MM:SS":
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{h}:{m:02d}:{sec:02d}"
        except Exception:
            return s

    @classmethod
    def clean_currency(cls, val: Any, null_placeholder: Optional[float] = None) -> Optional[float]:
        """Strip currency symbols, commas, and currency labels (PKR, Rs., $, etc.), returning clean float."""
        if val is None or pd.isna(val):
            return null_placeholder
        if isinstance(val, (int, float)):
            return float(val)
        
        s = str(val).strip()
        if not s or s.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return null_placeholder

        # Strip explicit currency words first
        s_clean = re.sub(r'(?i)\b(pkr|rs\.?|usd|eur|gbp|inr|cents?|dollars?)\b', '', s)
        # Strip currency symbols
        s_clean = re.sub(r'[\$\€\£\¥\₹]', '', s_clean)
        # Strip commas and any leading/trailing non-numeric except minus
        s_clean = s_clean.replace(',', '').strip()
        
        # Match valid float or int pattern
        match = re.search(r'[-+]?\d*\.?\d+', s_clean)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return null_placeholder
        return null_placeholder

    @classmethod
    def clean_boolean(cls, val: Any) -> Optional[bool]:
        """Convert diverse truthy/falsy representations to standard bool."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in ['true', '1', 'yes', 'y', 't', 'active', 'pass', 'enable', 'enabled']:
            return True
        elif s in ['false', '0', 'no', 'n', 'f', 'inactive', 'fail', 'disable', 'disabled']:
            return False
        return None

    @classmethod
    def clean_text(
        cls, 
        val: Any, 
        casing: Optional[str] = None, 
        strip_whitespace: bool = True
    ) -> Optional[str]:
        """Clean string values, sanitize whitespace, remove control characters and apply casing."""
        if val is None or pd.isna(val):
            return None
        s = str(val)
        if strip_whitespace:
            s = re.sub(r'\s+', ' ', s).strip()
        
        # Remove non-printable control characters
        s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)

        if casing == 'lower':
            s = s.lower()
        elif casing == 'upper':
            s = s.upper()
        elif casing == 'title':
            s = s.title()
        
        return s if s else None
