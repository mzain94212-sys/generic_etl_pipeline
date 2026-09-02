"""End-to-End Pipeline Integration Test."""

import os
from core.pipeline import GenericETLPipeline
from core.schemas import SemanticType


def test_e2e_pakistan_customers_pipeline():
    pipeline = GenericETLPipeline()
    csv_path = "/home/muhammad-zain/generic_etl_pipeline/data/samples/pakistan_customers_dirty.csv"
    
    result = pipeline.run(csv_path, filename="pakistan_customers_dirty.csv")
    
    assert result.success is True, f"Pipeline failed: {result.error_message}"
    assert result.extracted_rows == 9
    # Duplicate row should be removed -> 8 cleaned rows
    assert result.cleaned_rows == 8
    assert result.indexed_rows == 8
    
    # Check profile
    profile = result.dataset_profile
    assert profile is not None
    
    mob_prof = profile.columns["mob_number"]
    assert mob_prof.detected_type == SemanticType.PHONE_PAKISTAN
    assert mob_prof.confidence_score >= 0.80

    cnic_prof = profile.columns["nic_number"]
    assert cnic_prof.detected_type == SemanticType.CNIC_PAKISTAN

    salary_prof = profile.columns["salary_pkr"]
    assert salary_prof.detected_type == SemanticType.CURRENCY_AMOUNT

    age_prof = profile.columns["patient_age"]
    assert age_prof.detected_type == SemanticType.AGE

    # Check validation report
    val = result.validation_report
    assert val is not None
    assert val.valid_count > 0

    print("E2E Pakistan Customers Pipeline run succeeded!")
    print(f"Summary: Extracted {result.extracted_rows} -> Cleaned {result.cleaned_rows} -> Indexed {result.indexed_rows}")
    print(f"Execution time: {result.execution_time_seconds}s")
    for log in result.logs:
        print(f"  - {log}")


def test_e2e_test_csv_duration_pipeline():
    pipeline = GenericETLPipeline()
    csv_path = "/home/muhammad-zain/generic_etl_pipeline/test.csv"
    
    result = pipeline.run(csv_path, filename="test.csv")
    
    assert result.success is True, f"Pipeline failed: {result.error_message}"
    assert result.extracted_rows == 8
    assert result.cleaned_rows == 8
    
    profile = result.dataset_profile
    assert profile is not None
    
    # Verify duration column detected
    dur_prof = profile.columns["duration"]
    assert dur_prof.detected_type in [SemanticType.DURATION, SemanticType.TIME]
    assert dur_prof.confidence_score >= 0.80

    # Extract cleaned data to verify '12:36:53 AM' converted to '0:36:53'
    df_raw, _ = pipeline.extract(csv_path, filename="test.csv")
    cfg = pipeline.generate_default_cleaning_config(profile)
    df_clean, val, _ = pipeline.clean_and_transform(df_raw, cfg)

    # Row 0 duration was '12:36:53 AM' -> must be '0:36:53'
    assert df_clean.loc[0, "duration"] == "0:36:53"
    # Row 1 duration was '01:15:30 PM' -> '13:15:30'
    assert df_clean.loc[1, "duration"] == "13:15:30"
    # Row 2 duration was '12:00:00 AM' -> '0:00:00'
    assert df_clean.loc[2, "duration"] == "0:00:00"
    # Row 4 duration was '12:36:53 AM' -> '0:36:53'
    assert df_clean.loc[4, "duration"] == "0:36:53"
    # Row 7 duration was '11:59:59 PM' -> '23:59:59'
    assert df_clean.loc[7, "duration"] == "23:59:59"

    # Validation must pass
    assert val.passed is True


if __name__ == "__main__":
    test_e2e_pakistan_customers_pipeline()
    test_e2e_test_csv_duration_pipeline()
    print("All E2E Pipeline tests passed successfully!")
