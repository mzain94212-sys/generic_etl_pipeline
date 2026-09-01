"""Unit tests for semantic type detection and pattern inspection."""

import pytest
from core.schemas import SemanticType
from core.detector.semantic_matcher import SemanticColumnMatcher, fuzzy_score
from core.detector.pattern_engine import PatternEngine
from core.detector.type_inference import GenericTypeDetector


def test_pakistan_phone_detection():
    # Valid Pakistani numbers
    assert PatternEngine.test_pakistan_phone("03001234567") is True
    assert PatternEngine.test_pakistan_phone("0321-9876543") is True
    assert PatternEngine.test_pakistan_phone("+92 333 5551234") is True
    assert PatternEngine.test_pakistan_phone("00923451122334") is True
    assert PatternEngine.test_pakistan_phone("3001234567") is True
    assert PatternEngine.test_pakistan_phone("92-300-1234567") is True
    
    # Invalid numbers
    assert PatternEngine.test_pakistan_phone("12345") is False
    assert PatternEngine.test_pakistan_phone("02135551234") is False  # Landline non-3xx
    assert PatternEngine.test_pakistan_phone("abcde") is False


def test_pakistan_cnic_detection():
    assert PatternEngine.test_pakistan_cnic("35201-1234567-1") is True
    assert PatternEngine.test_pakistan_cnic("4210155443329") is True
    assert PatternEngine.test_pakistan_cnic("12345") is False


def test_fuzzy_column_matcher():
    matches = SemanticColumnMatcher.match_column_name("mob_number")
    assert matches[0][0] in [SemanticType.PHONE_PAKISTAN, SemanticType.PHONE_INTERNATIONAL]
    assert matches[0][1] > 0.60

    matches_cnic = SemanticColumnMatcher.match_column_name("shanakhti_card")
    assert matches_cnic[0][0] == SemanticType.CNIC_PAKISTAN

    matches_dob = SemanticColumnMatcher.match_column_name("d_o_b")
    assert matches_dob[0][0] == SemanticType.DATE


def test_email_and_datetime_patterns():
    assert PatternEngine.test_email("zain@example.com") is True
    assert PatternEngine.test_email("not_an_email") is False

    is_dt, _ = PatternEngine.test_datetime("2024-01-15")
    assert is_dt is True
    is_dt2, _ = PatternEngine.test_datetime("15/08/2023 14:30:00")
    assert is_dt2 is True


if __name__ == "__main__":
    test_pakistan_phone_detection()
    test_pakistan_cnic_detection()
    test_fuzzy_column_matcher()
    test_email_and_datetime_patterns()
    print("All Detector tests passed successfully!")
