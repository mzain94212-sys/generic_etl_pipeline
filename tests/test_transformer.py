"""Unit tests for transformer standardizers, missing value handler, and outlier handler."""

from core.transformer.normalizer import GenericNormalizer


def test_pakistan_phone_normalizer():
    # 03001234567 -> 923001234567
    assert GenericNormalizer.clean_pakistan_phone("03001234567", prefix="92") == "923001234567"
    assert GenericNormalizer.clean_pakistan_phone("+92 300 1234567", prefix="92") == "923001234567"
    assert GenericNormalizer.clean_pakistan_phone("00923001234567", prefix="+92") == "+923001234567"
    assert GenericNormalizer.clean_pakistan_phone("3001234567", prefix="92") == "923001234567"
    assert GenericNormalizer.clean_pakistan_phone("0321-9876543", prefix="92") == "923219876543"
    
    # Null & invalid handling
    assert GenericNormalizer.clean_pakistan_phone(None) is None
    assert GenericNormalizer.clean_pakistan_phone("nan") is None
    assert GenericNormalizer.clean_pakistan_phone("") is None


def test_datetime_and_currency_normalizer():
    # Datetime to standard ISO
    dt_str = GenericNormalizer.clean_datetime("2024/01/15 10:30:00", target_format="%Y-%m-%d %H:%M:%S")
    assert dt_str == "2024-01-15 10:30:00"

    # Currency stripping
    assert GenericNormalizer.clean_currency("PKR 145,000") == 145000.0
    assert GenericNormalizer.clean_currency("$2,500.50") == 2500.50
    assert GenericNormalizer.clean_currency("Rs. 95000") == 95000.0
    assert GenericNormalizer.clean_currency(None) is None


def test_cnic_normalizer():
    assert GenericNormalizer.clean_pakistan_cnic("3520112345671") == "35201-1234567-1"
    assert GenericNormalizer.clean_pakistan_cnic("35201-1234567-1") == "35201-1234567-1"
    assert GenericNormalizer.clean_pakistan_cnic(None) is None


def test_duration_normalizer():
    # 12-hour AM/PM to 24-hr duration (e.g., 12:36:53 AM -> 0:36:53)
    assert GenericNormalizer.clean_duration("12:36:53 AM") == "0:36:53"
    assert GenericNormalizer.clean_duration("12:36:53 PM") == "12:36:53"
    assert GenericNormalizer.clean_duration("01:15:30 AM") == "1:15:30"
    assert GenericNormalizer.clean_duration("01:15:30 PM") == "13:15:30"
    assert GenericNormalizer.clean_duration("12:00:00 AM") == "0:00:00"
    assert GenericNormalizer.clean_duration("12:00:00 PM") == "12:00:00"
    assert GenericNormalizer.clean_duration("11:59:59 PM") == "23:59:59"

    # Standard duration / 24-hr formats
    assert GenericNormalizer.clean_duration("0:36:53") == "0:36:53"
    assert GenericNormalizer.clean_duration("00:36:53") == "0:36:53"
    assert GenericNormalizer.clean_duration("14:20:00") == "14:20:00"

    # With HH:MM:SS target format
    assert GenericNormalizer.clean_duration("12:36:53 AM", target_format="HH:MM:SS") == "00:36:53"
    assert GenericNormalizer.clean_duration("01:15:30 AM", target_format="HH:MM:SS") == "01:15:30"

    # Text duration & numeric seconds
    assert GenericNormalizer.clean_duration("36m 53s") == "0:36:53"
    assert GenericNormalizer.clean_duration("2213") == "0:36:53"

    # Nulls & empty values
    assert GenericNormalizer.clean_duration(None) is None
    assert GenericNormalizer.clean_duration("nan") is None
    assert GenericNormalizer.clean_duration("") is None


if __name__ == "__main__":
    test_pakistan_phone_normalizer()
    test_datetime_and_currency_normalizer()
    test_cnic_normalizer()
    test_duration_normalizer()
    print("All Transformer tests passed successfully!")
