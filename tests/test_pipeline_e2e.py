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


if __name__ == "__main__":
    test_e2e_pakistan_customers_pipeline()
