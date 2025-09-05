# scripts/process_enrollment.py
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class EnrollmentProcessor(IPEDSProcessor):
    """Process EF2023 series - Fall Enrollment data."""

    def process(self) -> pd.DataFrame:
        """Process enrollment data from multiple EF files."""
        self.logger.info("Starting enrollment data processing...")

        # Load multiple enrollment files
        enrollment_files = {
            "ef2023a": "Fall enrollment by race/ethnicity and gender",
            "ef2023b": "Fall enrollment by age and gender",
            "ef2023c": "Fall enrollment by residence and migration",
        }

        processed_dfs = []

        for file_key, description in enrollment_files.items():
            try:
                df = self.load_csv(f"{file_key}.csv")
                self.logger.info(f"Processing {description}")

                if file_key == "ef2023a":
                    processed_df = self._process_race_ethnicity_enrollment(df)
                elif file_key == "ef2023b":
                    processed_df = self._process_age_enrollment(df)
                elif file_key == "ef2023c":
                    processed_df = self._process_residence_enrollment(df)

                processed_dfs.append(processed_df)

            except Exception as e:
                self.logger.warning(f"Could not process {file_key}: {e}")
                continue

        # Merge all enrollment data
        if processed_dfs:
            final_df = processed_dfs[0]
            for df in processed_dfs[1:]:
                final_df = final_df.merge(df, on="UNITID", how="outer")
        else:
            raise ValueError("No enrollment files could be processed")

        # Add overall derived fields
        final_df = self.add_derived_fields(final_df)

        # Validate and save
        validation_info = self.validate_data(final_df)
        self.save_processed_data(final_df, "enrollment_processed.csv", validation_info)

        return final_df

    def _process_race_ethnicity_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process race/ethnicity enrollment data."""

        # Key columns for race/ethnicity data
        race_columns = [
            "UNITID",
            "EFRACE01",
            "EFRACE02",
            "EFRACE03",
            "EFRACE04",
            "EFRACE05",  # Men by race
            "EFRACE06",
            "EFRACE07",
            "EFRACE08",
            "EFRACE09",
            "EFRACE10",  # Women by race
            "EFRACE11",
            "EFRACE12",
            "EFRACE13",
            "EFRACE14",
            "EFRACE15",  # Total by race
            "EFTOTLT",  # Grand total
        ]

        available_cols = [col for col in race_columns if col in df.columns]
        df_race = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_race = self.clean_numeric_columns(df_race, numeric_cols)

        # Calculate diversity metrics
        if "EFTOTLT" in df_race.columns:
            race_total_cols = [
                col
                for col in ["EFRACE11", "EFRACE12", "EFRACE13", "EFRACE14", "EFRACE15"]
                if col in df_race.columns
            ]

            if race_total_cols:
                # Calculate percentages by race (approximate mapping based on typical IPEDS structure)
                df_race["total_enrollment"] = df_race["EFTOTLT"]

                # Assuming standard IPEDS race categories
                race_mapping = {
                    "EFRACE11": "nonresident_alien_pct",
                    "EFRACE12": "hispanic_latino_pct",
                    "EFRACE13": "american_indian_alaska_native_pct",
                    "EFRACE14": "asian_pct",
                    "EFRACE15": "black_african_american_pct",
                }

                for col, pct_col in race_mapping.items():
                    if col in df_race.columns:
                        df_race[pct_col] = (
                            df_race[col] / df_race["total_enrollment"] * 100
                        ).round(2)

                # Calculate diversity index (1 - sum of squared proportions)
                diversity_cols = [
                    col for col in race_mapping.values() if col in df_race.columns
                ]
                if diversity_cols:
                    proportions = (
                        df_race[diversity_cols] / 100
                    )  # Convert percentages to proportions
                    df_race["diversity_index"] = (
                        1 - (proportions**2).sum(axis=1)
                    ).round(3)

        return df_race

    def _process_age_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process age enrollment data."""

        # Key age columns (typical IPEDS age categories)
        age_columns = [
            "UNITID",
            "EFAGE01",
            "EFAGE02",
            "EFAGE03",
            "EFAGE04",
            "EFAGE05",  # Men by age
            "EFAGE06",
            "EFAGE07",
            "EFAGE08",
            "EFAGE09",
            "EFAGE10",  # Women by age
            "EFAGE11",
            "EFAGE12",
            "EFAGE13",
            "EFAGE14",
            "EFAGE15",  # Total by age
        ]

        available_cols = [col for col in age_columns if col in df.columns]
        df_age = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_age = self.clean_numeric_columns(df_age, numeric_cols)

        # Calculate age distribution metrics
        total_cols = [
            col
            for col in ["EFAGE11", "EFAGE12", "EFAGE13", "EFAGE14", "EFAGE15"]
            if col in df_age.columns
        ]

        if total_cols:
            df_age["total_enrollment_age"] = df_age[total_cols].sum(axis=1)

            # Age category mapping (approximate based on typical IPEDS structure)
            age_mapping = {
                "EFAGE11": "under_18_pct",
                "EFAGE12": "age_18_19_pct",
                "EFAGE13": "age_20_21_pct",
                "EFAGE14": "age_22_24_pct",
                "EFAGE15": "age_25_over_pct",
            }

            for col, pct_col in age_mapping.items():
                if col in df_age.columns:
                    df_age[pct_col] = (
                        df_age[col] / df_age["total_enrollment_age"] * 100
                    ).round(2)

            # Calculate traditional vs non-traditional student percentages
            traditional_cols = [
                col
                for col in ["EFAGE11", "EFAGE12", "EFAGE13"]
                if col in df_age.columns
            ]
            nontraditional_cols = [
                col for col in ["EFAGE14", "EFAGE15"] if col in df_age.columns
            ]

            if traditional_cols:
                df_age["traditional_age_students_pct"] = (
                    df_age[traditional_cols].sum(axis=1)
                    / df_age["total_enrollment_age"]
                    * 100
                ).round(2)

            if nontraditional_cols:
                df_age["nontraditional_age_students_pct"] = (
                    df_age[nontraditional_cols].sum(axis=1)
                    / df_age["total_enrollment_age"]
                    * 100
                ).round(2)

        return df_age

    def _process_residence_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process residence and migration enrollment data."""

        # Key residence columns
        residence_columns = [
            "UNITID",
            "EFRES01",
            "EFRES02",
            "EFRES03",  # Men: in-state, out-of-state, foreign
            "EFRES04",
            "EFRES05",
            "EFRES06",  # Women: in-state, out-of-state, foreign
            "EFRES07",
            "EFRES08",
            "EFRES09",  # Total: in-state, out-of-state, foreign
        ]

        available_cols = [col for col in residence_columns if col in df.columns]
        df_res = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_res = self.clean_numeric_columns(df_res, numeric_cols)

        # Calculate residence distribution
        total_cols = [
            col for col in ["EFRES07", "EFRES08", "EFRES09"] if col in df_res.columns
        ]

        if total_cols:
            df_res["total_enrollment_residence"] = df_res[total_cols].sum(axis=1)

            # Residence mapping
            residence_mapping = {
                "EFRES07": "in_state_pct",
                "EFRES08": "out_of_state_pct",
                "EFRES09": "international_pct",
            }

            for col, pct_col in residence_mapping.items():
                if col in df_res.columns:
                    df_res[pct_col] = (
                        df_res[col] / df_res["total_enrollment_residence"] * 100
                    ).round(2)

            # Geographic diversity indicator
            if (
                "out_of_state_pct" in df_res.columns
                and "international_pct" in df_res.columns
            ):
                df_res["geographic_diversity_pct"] = (
                    df_res["out_of_state_pct"] + df_res["international_pct"]
                ).round(2)

        return df_res

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall derived fields for enrollment data."""
        df = df.copy()

        # Determine primary enrollment total
        enrollment_cols = [
            "total_enrollment",
            "total_enrollment_age",
            "total_enrollment_residence",
        ]
        available_enrollment_cols = [
            col for col in enrollment_cols if col in df.columns
        ]

        if available_enrollment_cols:
            # Use the first available total enrollment column
            df["student_body_size"] = df[available_enrollment_cols[0]]

        # Create size categories
        if "student_body_size" in df.columns:

            def categorize_size(size):
                if pd.isna(size) or size == 0:
                    return "Unknown"
                elif size < 1000:
                    return "Very Small (<1,000)"
                elif size < 3000:
                    return "Small (1,000-2,999)"
                elif size < 10000:
                    return "Medium (3,000-9,999)"
                elif size < 20000:
                    return "Large (10,000-19,999)"
                else:
                    return "Very Large (20,000+)"

            df["enrollment_size_category"] = df["student_body_size"].apply(
                categorize_size
            )

        # Diversity flags
        if "diversity_index" in df.columns:
            df["highly_diverse"] = (df["diversity_index"] >= 0.6).astype(int)
            df["low_diversity"] = (df["diversity_index"] <= 0.3).astype(int)

        # Geographic reach flags
        if "geographic_diversity_pct" in df.columns:
            df["regional_draw"] = (df["geographic_diversity_pct"] <= 25).astype(int)
            df["national_draw"] = (df["geographic_diversity_pct"] >= 50).astype(int)

        # Non-traditional student friendly
        if "nontraditional_age_students_pct" in df.columns:
            df["nontraditional_friendly"] = (
                df["nontraditional_age_students_pct"] >= 25
            ).astype(int)

        # International presence
        if "international_pct" in df.columns:
            df["significant_international_presence"] = (
                df["international_pct"] >= 5
            ).astype(int)

        return df


if __name__ == "__main__":
    processor = EnrollmentProcessor()
    processed_df = processor.process()
    print(f"\nProcessed enrollment data for {len(processed_df)} institutions")
    print("\nSample processed data:")
    sample_cols = [
        "UNITID",
        "student_body_size",
        "enrollment_size_category",
        "diversity_index",
        "geographic_diversity_pct",
    ]
    available_sample_cols = [col for col in sample_cols if col in processed_df.columns]
    print(processed_df[available_sample_cols].head())
