"""
Deep Data Pattern and Structural Value Inspection Engine.
Inspects raw column sample values using regular expressions, statistical heuristics,
and domain-specific validators (PK phone numbers, CNIC, datetimes, emails, IP, currency, etc.).
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import dateutil.parser
from ..schemas import SemanticType


class PatternEngine:
    """Evaluates raw sample values against regex patterns and domain-specific rules."""

    RE_PAKISTAN_PHONE = re.compile(r'^(?:\+92|0092|92|0)?[- ]?(3\d{2})[- ]?(\d{7})$')
    RE_INTERNATIONAL_PHONE = re.compile(r'^\+(?:[0-9] ?){6,14}[0-9]$')
    RE_PAKISTAN_CNIC = re.compile(r'^\d{5}[- ]?\d{7}[- ]?\d{1}$')
    RE_EMAIL = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    RE_CURRENCY = re.compile(r'^(?:[\$\€\£\¥\₹]|PKR|Rs\.?|USD|EUR|GBP|INR)?\s*[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:PKR|Rs\.?|USD|EUR)?$', re.IGNORECASE)
    RE_PERCENTAGE = re.compile(r'^[-+]?[0-9]+(?:\.[0-9]+)?\s*%$')
    RE_IPV4 = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    RE_IPV6 = re.compile(r'^(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}$', re.IGNORECASE)
    RE_URL = re.compile(r'^(?:https?:\/\/|ftp:\/\/|www\.)[^\s\/$.?#].[^\s]*$', re.IGNORECASE)
    RE_TIME_12H = re.compile(r'^\d{1,2}:\d{2}(?::\d{2})?\s*[ap]m$', re.IGNORECASE)
    RE_TIME_24H = re.compile(r'^\d{1,3}:\d{2}(?::\d{2})?$')
    RE_DURATION_TEXT = re.compile(r'^(?:\d+\s*(?:h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)\s*)+$', re.IGNORECASE)
    RE_ADDRESS_KEYWORDS = re.compile(r'\b(street|st|road|rd|avenue|ave|lane|ln|sector|block|phase|house|h#|flat|apt|plot|mohallah|colony|dha|bahria)\b', re.IGNORECASE)

    BOOLEAN_VALUES = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f', 'active', 'inactive', 'pass', 'fail'}
    GENDER_VALUES = {'male', 'female', 'm', 'f', 'other', 'non-binary', 'mard', 'aurat', 'transgender'}

    @classmethod
    def test_pakistan_phone(cls, val: str) -> bool:
        """Check if value is a valid Pakistani mobile number (03xx / 923xx / +923xx)."""
        clean = re.sub(r'[\s\-()]', '', val)
        return bool(cls.RE_PAKISTAN_PHONE.match(clean))

    @classmethod
    def test_international_phone(cls, val: str) -> bool:
        """Check if value is a valid international phone number starting with +."""
        val_s = val.strip()
        if val_s.startswith('+') and bool(cls.RE_INTERNATIONAL_PHONE.match(val_s)):
            return True
        return False

    @classmethod
    def test_pakistan_cnic(cls, val: str) -> bool:
        """Check if value is a 13-digit Pakistani CNIC."""
        val_s = val.strip()
        clean = re.sub(r'[\s\-]', '', val_s)
        if len(clean) == 13 and clean.isdigit():
            # Check if has standard 5-7-1 format or plain 13 digits
            return True
        return bool(cls.RE_PAKISTAN_CNIC.match(val_s))

    @classmethod
    def test_email(cls, val: str) -> bool:
        """Check if value matches standard email syntax."""
        return bool(cls.RE_EMAIL.match(val.strip()))

    @classmethod
    def test_datetime(cls, val: str) -> Tuple[bool, Optional[str]]:
        """Check if string can be parsed as a datetime."""
        val_s = val.strip()
        if not val_s:
            return False, None
        
        # Check numeric epoch timestamp (10 or 13 digits)
        if val_s.isdigit() and len(val_s) in [10, 13]:
            return True, "epoch"

        # Must have date separators (- / : T .) or words
        if not any(c in val_s for c in ['-', '/', ':', 'T', ' ']):
            return False, None

        if len(val_s) < 5 or val_s.isdigit():
            return False, None

        try:
            parsed = dateutil.parser.parse(val_s, fuzzy=False)
            return True, "standard_datetime"
        except (ValueError, OverflowError, TypeError):
            return False, None

    @classmethod
    def test_time_or_duration(cls, val: str) -> bool:
        """Check if value matches standard time (12h AM/PM or 24h) or duration syntax."""
        val_s = val.strip()
        if not val_s:
            return False
        # Disqualify if full date indicators are present
        if '-' in val_s or '/' in val_s:
            return False
        if bool(cls.RE_TIME_12H.match(val_s)):
            return True
        if bool(cls.RE_TIME_24H.match(val_s)):
            return True
        if bool(cls.RE_DURATION_TEXT.match(val_s)):
            return True
        return False

    @classmethod
    def test_currency(cls, val: str) -> bool:
        """Check if value contains currency symbols or explicit currency text."""
        val_s = val.strip()
        has_symbol = any(s in val_s for s in ['$', '€', '£', '¥', '₹', 'PKR', 'Rs', 'Rs.', 'pkr', 'rs'])
        if has_symbol:
            cleaned = re.sub(r'[^\d.-]', '', val_s)
            try:
                float(cleaned)
                return True
            except ValueError:
                return False
        return False

    @classmethod
    def test_percentage(cls, val: str) -> bool:
        """Check if value is a formatted percentage."""
        return bool(cls.RE_PERCENTAGE.match(val.strip()))

    @classmethod
    def test_ip_address(cls, val: str) -> bool:
        """Check if value is IPv4 or IPv6."""
        val_s = val.strip()
        return bool(cls.RE_IPV4.match(val_s) or cls.RE_IPV6.match(val_s))

    @classmethod
    def test_url(cls, val: str) -> bool:
        """Check if value is a valid URL."""
        return bool(cls.RE_URL.match(val.strip()))

    @classmethod
    def test_age(cls, val: Any) -> bool:
        """Check if value represents a realistic human age (0 to 120)."""
        try:
            v = float(val)
            return v.is_integer() and 0 <= v <= 120
        except (ValueError, TypeError):
            return False

    @classmethod
    def test_address(cls, val: str) -> bool:
        """Check if text looks like a postal or street address."""
        val_s = val.strip()
        if len(val_s) > 10 and bool(cls.RE_ADDRESS_KEYWORDS.search(val_s)):
            return True
        return False

    @classmethod
    def evaluate_sample_values(cls, sample_values: List[Any]) -> Dict[SemanticType, float]:
        """
        Evaluate sample values across all pattern detectors.
        Returns a dict of {SemanticType: match_ratio (0.0 to 1.0)}.
        """
        non_null_samples = [str(v).strip() for v in sample_values if v is not None and str(v).strip() != '' and str(v).lower() != 'nan' and str(v).lower() != 'null']
        total = len(non_null_samples)
        if total == 0:
            return {SemanticType.UNKNOWN: 1.0}

        counts: Dict[SemanticType, int] = {t: 0 for t in SemanticType}

        for val in non_null_samples:
            # 1. Pakistani Phone
            if cls.test_pakistan_phone(val):
                counts[SemanticType.PHONE_PAKISTAN] += 1
            
            # 2. CNIC (Check before international phone!)
            elif cls.test_pakistan_cnic(val):
                counts[SemanticType.CNIC_PAKISTAN] += 1

            # 3. International Phone
            elif cls.test_international_phone(val):
                counts[SemanticType.PHONE_INTERNATIONAL] += 1

            # 4. Email
            elif cls.test_email(val):
                counts[SemanticType.EMAIL] += 1

            # 5. Time & Duration (check before generic datetime)
            elif cls.test_time_or_duration(val):
                counts[SemanticType.DURATION] += 1
                counts[SemanticType.TIME] += 1

            # 6. DateTime
            elif True:
                is_dt, _ = cls.test_datetime(val)
                if is_dt:
                    counts[SemanticType.DATETIME] += 1
                elif cls.test_currency(val):
                    counts[SemanticType.CURRENCY_AMOUNT] += 1
                elif cls.test_percentage(val):
                    counts[SemanticType.PERCENTAGE] += 1
                elif cls.test_ip_address(val):
                    counts[SemanticType.IP_ADDRESS] += 1
                elif cls.test_url(val):
                    counts[SemanticType.URL] += 1
                elif val.lower() in cls.BOOLEAN_VALUES:
                    counts[SemanticType.BOOLEAN] += 1
                elif val.lower() in cls.GENDER_VALUES:
                    counts[SemanticType.GENDER] += 1
                elif cls.test_age(val):
                    counts[SemanticType.AGE] += 1
                elif cls.test_address(val):
                    counts[SemanticType.ADDRESS] += 1
                elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    counts[SemanticType.NUMERIC_INTEGER] += 1
                else:
                    try:
                        float(val)
                        counts[SemanticType.NUMERIC_FLOAT] += 1
                    except ValueError:
                        pass

        # Compute ratios
        ratios = {sem_type: count / total for sem_type, count in counts.items() if count > 0}
        return ratios
