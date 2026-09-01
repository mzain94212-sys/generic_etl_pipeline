"""
Semantic Column Name Matcher.
Uses fuzzy string matching, token sorting, and a multilingual/domain-specific synonym ontology
to infer semantic intent solely from column names.
"""

import re
from typing import Dict, List, Optional, Tuple
from ..schemas import SemanticType

# Optional RapidFuzz import with pure Python fallback
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Pure Python Levenshtein similarity ratio (0.0 to 1.0) fallback."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )
    dist = matrix[len1][len2]
    return 1.0 - (dist / max(len1, len2))


def fuzzy_score(query: str, target: str) -> float:
    """Compute fuzzy match similarity between 0.0 and 1.0."""
    q = query.lower().strip()
    t = target.lower().strip()
    if q == t:
        return 1.0
    if q in t or t in q:
        return max(0.85, len(min(q, t, key=len)) / len(max(q, t, key=len)))
    
    if HAS_RAPIDFUZZ:
        ratio = fuzz.token_sort_ratio(q, t) / 100.0
        partial = fuzz.partial_ratio(q, t) / 100.0
        return max(ratio, partial * 0.9)
    else:
        return _levenshtein_ratio(q, t)


class SemanticColumnMatcher:
    """Matches column names against an extensive synonym dictionary of semantic data types."""

    ONTOLOGY: Dict[SemanticType, List[str]] = {
        # Contact & Identity
        SemanticType.PHONE_PAKISTAN: [
            "phone", "mobile", "cell", "cellphone", "mob", "ph_no", "phone_number",
            "mobile_no", "contact_no", "contact_number", "whatsapp", "wa_number",
            "rabta_no", "sim_number", "mobile_phone", "tele_no", "call_number",
            "pk_phone", "pakistan_number", "pak_phone", "jazz_no", "telenor_no", "ufone_no", "zong_no", "mob_number"
        ],
        SemanticType.PHONE_INTERNATIONAL: [
            "intl_phone", "international_phone", "work_phone", "office_phone",
            "fax", "telephone", "landline", "emergency_phone", "contact_intl"
        ],
        SemanticType.EMAIL: [
            "email", "e_mail", "mail", "user_email", "contact_email", "email_address",
            "emailaddress", "work_email", "personal_email", "customer_email"
        ],
        SemanticType.CNIC_PAKISTAN: [
            "cnic", "nic", "national_id", "shanakhti_card", "shanakhticard", "id_card",
            "identity_number", "pk_cnic", "nadra_id", "cnic_number", "nic_no", "cnic_no", "nic_number"
        ],
        SemanticType.CITY: [
            "city", "city_name", "town", "district", "municipality", "shehr", "shipping_city", "billing_city", "customer_city"
        ],
        SemanticType.COUNTRY: [
            "country", "country_name", "nation", "nationality", "mulk", "shipping_country", "country_code"
        ],
        SemanticType.NAME_PERSON: [
            "full_name", "first_name", "last_name", "fname", "lname",
            "customer_name", "client_name", "patient_name", "employee_name",
            "user_name", "username", "contact_name", "student_name", "person_name", "naam", "name"
        ],
        SemanticType.AGE: [
            "age", "user_age", "patient_age", "years_old", "age_years", "umr", "customer_age"
        ],
        SemanticType.GENDER: [
            "gender", "sex", "jins", "male_female", "user_gender"
        ],
        SemanticType.ADDRESS: [
            "address", "addr", "street", "shipping_address", "billing_address",
            "residential_address", "delivery_address", "street_address", "home_address", "pata"
        ],
        SemanticType.POSTAL_CODE: [
            "postal_code", "zip", "zipcode", "postcode", "pin_code", "zip_code"
        ],

        # Temporal
        SemanticType.DATETIME: [
            "datetime", "timestamp", "created_at", "updated_at", "created_date",
            "modified_date", "registered_at", "registered_datetime", "log_time", "event_time", "order_datetime",
            "start_time", "end_time", "occurred_at"
        ],
        SemanticType.DATE: [
            "date", "dob", "d_o_b", "date_of_birth", "birthdate", "birth_date", "expiry_date",
            "exp_date", "order_date", "join_date", "joined_on", "trans_date", "due_date",
            "effective_date", "hire_date", "tarikh"
        ],
        SemanticType.TIME: [
            "time", "time_of_day", "hh_mm_ss", "punch_time", "check_in_time", "check_out_time", "waqt"
        ],
        SemanticType.YEAR: [
            "year", "yr", "birth_year", "model_year", "fiscal_year", "sal", "academic_year"
        ],
        SemanticType.TIMESTAMP_EPOCH: [
            "epoch", "epoch_time", "unix_timestamp", "epoch_ms", "ts_epoch"
        ],

        # Financial & Numeric
        SemanticType.CURRENCY_AMOUNT: [
            "salary", "salary_pkr", "price", "amount", "cost", "budget", "revenue", "fare", "fee",
            "balance", "total", "subtotal", "payment", "net_amount", "gross_amount",
            "tankhwah", "pkr", "usd", "unit_price", "total_price", "discount_amount"
        ],
        SemanticType.PERCENTAGE: [
            "percentage", "percent", "pct", "discount_pct", "tax_rate", "rate", "ratio",
            "growth_rate", "margin", "interest_rate"
        ],
        SemanticType.IDENTIFIER_ID: [
            "cust_id", "id", "uuid", "guid", "code", "customer_id", "user_id", "order_id",
            "txn_id", "transaction_id", "account_id", "tracking_no", "sku", "product_id"
        ],
        SemanticType.NUMERIC_INTEGER: [
            "count", "qty", "quantity", "num_items", "total_items", "score", "points",
            "rank", "index", "tadaad"
        ],
        SemanticType.NUMERIC_FLOAT: [
            "weight", "height", "temperature", "latitude", "longitude", "lat", "lng",
            "metric", "score_float", "average", "avg", "wazan"
        ],

        # Web & Network
        SemanticType.URL: ["url", "website", "link", "web_address", "href", "uri", "profile_url", "site"],
        SemanticType.IP_ADDRESS: ["ip", "ip_address", "ipv4", "ipv6", "client_ip", "remote_ip", "host_ip"],
        SemanticType.MAC_ADDRESS: ["mac", "mac_address", "hardware_address"],

        # Categorical & Logical
        SemanticType.BOOLEAN: [
            "is_active", "status_active", "enabled", "is_verified", "verified",
            "has_subscribed", "is_deleted", "flag", "opt_in", "success", "is_valid"
        ],
        SemanticType.CATEGORICAL: [
            "status", "category", "type", "tier", "grade", "department", "dept",
            "role", "priority", "level", "state", "maritial_status", "halat"
        ],

        # Text
        SemanticType.TEXT_LONG: [
            "description", "notes", "comment", "feedback", "review", "summary",
            "detail", "remarks", "message", "body", "bio", "tafseel"
        ],
    }

    @classmethod
    def match_column_name(cls, col_name: str) -> List[Tuple[SemanticType, float]]:
        """
        Match a raw column name against all known semantic types.
        Returns a sorted list of (SemanticType, similarity_score).
        """
        cleaned_col = re.sub(r'[^a-zA-Z0-9_]', '', col_name.lower().replace(' ', '_').replace('-', '_'))
        
        matches: List[Tuple[SemanticType, float]] = []

        for sem_type, synonyms in cls.ONTOLOGY.items():
            best_score = 0.0
            for syn in synonyms:
                score = fuzzy_score(cleaned_col, syn)
                if score > best_score:
                    best_score = score
            if best_score > 0.40:
                matches.append((sem_type, best_score))

        matches.sort(key=lambda x: x[1], reverse=True)
        if not matches:
            matches.append((SemanticType.UNKNOWN, 0.0))
        return matches
