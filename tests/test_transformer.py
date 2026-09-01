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


if __name__ == "__main__":
    test_pakistan_phone_normalizer()
    test_datetime_and_currency_normalizer()
    test_cnic_normalizer()
    print("All Transformer tests passed successfully!")
