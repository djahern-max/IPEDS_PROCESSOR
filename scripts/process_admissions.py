# scripts/process_admissions.py
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class AdmissionsProcessor(IPEDSProcessor):
    """Process ADM2023 - Admissions and Test Scores."""

    def process(self) -> pd.DataFrame:
        """Process admissions data."""
        self.logger.info("Starting admissions data processing...")

        # Load raw data
        df = self.load_csv("adm2023.csv")

        # Key admissions columns
        key_columns = [
            "UNITID",  # Institution identifier
            "APPLCN",
            "APPLCNM",
            "APPLCNW",  # Applications received (total, men, women)
            "ADMSSN",
            "ADMSSNM",
            "ADMSSNW",  # Admissions (total, men, women)
            "ENRLT",
            "ENRLTM",
            "ENRLTW",  # Enrolled (total, men, women)
            "ENRLFT",
            "ENRLPT",  # Enrolled full-time, part-time
            "SATNUM",
            "SATPCT",  # SAT submissions
            "ACTNUM",
            "ACTPCT",  # ACT submissions
            "SATVR25",
            "SATVR75",  # SAT Evidence-Based Reading and Writing 25th/75th percentile
            "SATMT25",
            "SATMT75",  # SAT Math 25th/75th percentile
            "SATWR25",
            "SATWR75",  # SAT Writing 25th/75th percentile (if available)
            "ACTCM25",
            "ACTCM75",  # ACT Composite 25th/75th percentile
            "ACTEN25",
            "ACTEN75",  # ACT English 25th/75th percentile
            "ACTMT25",
            "ACTMT75",  # ACT Math 25th/75th percentile
            "ACTWR25",
            "ACTWR75",  # ACT Writing 25th/75th percentile
        ]

        # Filter to available columns
        available_columns = [col for col in key_columns if col in df.columns]
        df = df[available_columns].copy()

        # Clean numeric columns
        numeric_columns = [col for col in available_columns if col != "UNITID"]
        df = self.clean_numeric_columns(df, numeric_columns)

        # Add derived fields
        df = self.add_derived_fields(df)

        # Filter out institutions with no admissions data
        df = df[df["has_admissions_data"] == True].copy()

        # Validate and save
        validation_info = self.validate_data(df)
        self.save_processed_data(df, "admissions_processed.csv", validation_info)

        return df

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields for admissions analysis."""
        df = df.copy()

        # Calculate acceptance rate
        if "APPLCN" in df.columns and "ADMSSN" in df.columns:
            df["acceptance_rate"] = df["ADMSSN"] / df["APPLCN"] * 100
            df["acceptance_rate"] = df["acceptance_rate"].round(2)

        # Calculate yield rate (enrolled / admitted)
        if "ENRLT" in df.columns and "ADMSSN" in df.columns:
            df["yield_rate"] = df["ENRLT"] / df["ADMSSN"] * 100
            df["yield_rate"] = df["yield_rate"].round(2)

        # Calculate selectivity categories
        if "acceptance_rate" in df.columns:

            def categorize_selectivity(rate):
                if pd.isna(rate):
                    return "Unknown"
                elif rate <= 10:
                    return "Most competitive (≤10%)"
                elif rate <= 25:
                    return "Highly competitive (11-25%)"
                elif rate <= 50:
                    return "Competitive (26-50%)"
                elif rate <= 75:
                    return "Moderately competitive (51-75%)"
                else:
                    return "Less competitive (>75%)"

            df["selectivity_category"] = df["acceptance_rate"].apply(
                categorize_selectivity
            )

        # Calculate combined SAT scores
        if "SATVR25" in df.columns and "SATMT25" in df.columns:
            df["sat_total_25"] = df["SATVR25"] + df["SATMT25"]
        if "SATVR75" in df.columns and "SATMT75" in df.columns:
            df["sat_total_75"] = df["SATVR75"] + df["SATMT75"]

        # Calculate SAT/ACT score ranges
        if "sat_total_25" in df.columns and "sat_total_75" in df.columns:
            df["sat_range"] = (
                df["sat_total_25"].astype(str) + " - " + df["sat_total_75"].astype(str)
            )
            df["sat_range"] = df["sat_range"].replace("nan - nan", np.nan)

        if "ACTCM25" in df.columns and "ACTCM75" in df.columns:
            df["act_range"] = (
                df["ACTCM25"].astype(str) + " - " + df["ACTCM75"].astype(str)
            )
            df["act_range"] = df["act_range"].replace("nan - nan", np.nan)

        # Test score submission rates
        if "SATNUM" in df.columns and "ENRLT" in df.columns:
            df["sat_submission_rate"] = (df["SATNUM"] / df["ENRLT"] * 100).round(2)

        if "ACTNUM" in df.columns and "ENRLT" in df.columns:
            df["act_submission_rate"] = (df["ACTNUM"] / df["ENRLT"] * 100).round(2)

        # Gender distribution of applicants
        if (
            "APPLCNM" in df.columns
            and "APPLCNW" in df.columns
            and "APPLCN" in df.columns
        ):
            df["pct_male_applicants"] = (df["APPLCNM"] / df["APPLCN"] * 100).round(2)
            df["pct_female_applicants"] = (df["APPLCNW"] / df["APPLCN"] * 100).round(2)

        # Competitiveness flags based on test scores
        if "sat_total_75" in df.columns:
            df["highly_competitive_sat"] = (df["sat_total_75"] >= 1400).astype(int)

        if "ACTCM75" in df.columns:
            df["highly_competitive_act"] = (df["ACTCM75"] >= 32).astype(int)

        # Create flags for data availability
        df["has_admissions_data"] = (
            df[["APPLCN", "ADMSSN", "ENRLT"]].notna().any(axis=1)
        )
        df["has_sat_scores"] = (
            df[["SATVR25", "SATMT25"]].notna().all(axis=1)
            if "SATVR25" in df.columns
            else False
        )
        df["has_act_scores"] = (
            df[["ACTCM25"]].notna().any(axis=1) if "ACTCM25" in df.columns else False
        )

        # Test optional indicator (low submission rates might indicate test optional)
        test_optional_threshold = (
            25  # If less than 25% submit scores, likely test optional
        )
        df["likely_test_optional"] = False
        if "sat_submission_rate" in df.columns:
            df["likely_test_optional"] |= (
                df["sat_submission_rate"] < test_optional_threshold
            ) & df["sat_submission_rate"].notna()
        if "act_submission_rate" in df.columns:
            df["likely_test_optional"] |= (
                df["act_submission_rate"] < test_optional_threshold
            ) & df["act_submission_rate"].notna()

        return df


if __name__ == "__main__":
    processor = AdmissionsProcessor()
    processed_df = processor.process()
    print(f"\nProcessed {len(processed_df)} institutions with admissions data")
    print("\nSample processed data:")
    sample_cols = [
        "UNITID",
        "acceptance_rate",
        "selectivity_category",
        "sat_range",
        "act_range",
    ]
    available_sample_cols = [col for col in sample_cols if col in processed_df.columns]
    print(processed_df[available_sample_cols].head())
