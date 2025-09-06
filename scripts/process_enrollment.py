# Fixed version of process_enrollment.py
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class EnrollmentProcessor(IPEDSProcessor):
    """Fixed Process EF2023 series - Fall Enrollment data."""

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
        base_unitids = None

        for file_key, description in enrollment_files.items():
            try:
                df = self.load_csv(f"{file_key}.csv")
                self.logger.info(f"Processing {description}: {len(df)} rows")

                # CRITICAL FIX: Validate UNITIDs before processing
                if base_unitids is None:
                    base_unitids = set(df["UNITID"].unique())
                    self.logger.info(f"Base UNITID count: {len(base_unitids)}")
                else:
                    current_unitids = set(df["UNITID"].unique())
                    if len(current_unitids) != len(base_unitids):
                        self.logger.warning(
                            f"UNITID count mismatch: {len(current_unitids)} vs {len(base_unitids)}"
                        )

                if file_key == "ef2023a":
                    processed_df = self._process_race_ethnicity_enrollment(df)
                elif file_key == "ef2023b":
                    processed_df = self._process_age_enrollment(df)
                elif file_key == "ef2023c":
                    processed_df = self._process_residence_enrollment(df)

                # CRITICAL FIX: Validate processed data
                if len(processed_df) > len(base_unitids) * 1.1:  # Allow 10% tolerance
                    self.logger.error(
                        f"Data multiplication detected in {file_key}: {len(processed_df)} rows for {len(base_unitids)} institutions"
                    )
                    # Fix by removing duplicates
                    processed_df = processed_df.drop_duplicates(
                        subset=["UNITID"], keep="first"
                    )
                    self.logger.info(f"After deduplication: {len(processed_df)} rows")

                processed_dfs.append(processed_df)

            except Exception as e:
                self.logger.warning(f"Could not process {file_key}: {e}")
                continue

        # CRITICAL FIX: Merge properly with validation
        if processed_dfs:
            final_df = processed_dfs[0].copy()
            original_count = len(final_df)

            for i, df in enumerate(processed_dfs[1:], 1):
                before_merge = len(final_df)
                final_df = final_df.merge(df, on="UNITID", how="outer")
                after_merge = len(final_df)

                # Validate merge didn't create duplicates
                if (
                    after_merge > before_merge * 1.1
                ):  # More than 10% increase is suspicious
                    self.logger.error(
                        f"Merge {i} created unexpected row increase: {before_merge} -> {after_merge}"
                    )
                    # Remove duplicates
                    final_df = final_df.drop_duplicates(subset=["UNITID"], keep="first")
                    self.logger.info(f"After deduplication: {len(final_df)} rows")

                self.logger.info(f"Merge {i} complete: {len(final_df)} institutions")
        else:
            raise ValueError("No enrollment files could be processed")

        # Final validation
        final_unique_unitids = final_df["UNITID"].nunique()
        if final_unique_unitids != len(final_df):
            self.logger.error(
                f"Final dataset has duplicates: {len(final_df)} rows, {final_unique_unitids} unique UNITIDs"
            )
            final_df = final_df.drop_duplicates(subset=["UNITID"], keep="first")
            self.logger.info(f"Final dataset after deduplication: {len(final_df)} rows")

        # Add overall derived fields
        final_df = self.add_derived_fields(final_df)

        # Validate and save
        validation_info = self.validate_data(final_df)
        self.save_processed_data(final_df, "enrollment_processed.csv", validation_info)

        return final_df

    def _process_race_ethnicity_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process race/ethnicity enrollment data - FIXED VERSION."""

        # CRITICAL FIX: Only process unique UNITIDs
        original_count = len(df)
        df = df.drop_duplicates(subset=["UNITID"], keep="first")
        if len(df) != original_count:
            self.logger.warning(
                f"Removed {original_count - len(df)} duplicate UNITIDs in race/ethnicity data"
            )

        # Key columns for race/ethnicity data
        race_columns = [
            "UNITID",
            "EFTOTLT",  # Total enrollment
            # Add other columns as they exist in your data
        ]

        available_cols = [col for col in race_columns if col in df.columns]
        df_race = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_race = self.clean_numeric_columns(df_race, numeric_cols)

        # Simple total enrollment field
        if "EFTOTLT" in df_race.columns:
            df_race["total_enrollment"] = df_race["EFTOTLT"]

        return df_race

    def _process_age_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process age enrollment data - FIXED VERSION."""

        # CRITICAL FIX: Only process unique UNITIDs
        original_count = len(df)
        df = df.drop_duplicates(subset=["UNITID"], keep="first")
        if len(df) != original_count:
            self.logger.warning(
                f"Removed {original_count - len(df)} duplicate UNITIDs in age data"
            )

        # Key age columns - simplified
        age_columns = ["UNITID"]

        # Add age columns that actually exist
        for col in df.columns:
            if col.startswith("EFAGE") and col != "UNITID":
                age_columns.append(col)

        available_cols = [col for col in age_columns if col in df.columns]
        df_age = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_age = self.clean_numeric_columns(df_age, numeric_cols)

        return df_age

    def _process_residence_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process residence enrollment data - FIXED VERSION."""

        # CRITICAL FIX: Only process unique UNITIDs
        original_count = len(df)
        df = df.drop_duplicates(subset=["UNITID"], keep="first")
        if len(df) != original_count:
            self.logger.warning(
                f"Removed {original_count - len(df)} duplicate UNITIDs in residence data"
            )

        # Key residence columns - simplified
        residence_columns = ["UNITID"]

        # Add residence columns that actually exist
        for col in df.columns:
            if col.startswith("EFRES") and col != "UNITID":
                residence_columns.append(col)

        available_cols = [col for col in residence_columns if col in df.columns]
        df_res = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_res = self.clean_numeric_columns(df_res, numeric_cols)

        return df_res

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall derived fields for enrollment data."""
        df = df.copy()

        # Determine primary enrollment total
        enrollment_cols = ["total_enrollment", "EFTOTLT"]
        available_enrollment_cols = [
            col for col in enrollment_cols if col in df.columns
        ]

        if available_enrollment_cols:
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

        return df


if __name__ == "__main__":
    processor = EnrollmentProcessor()
    processed_df = processor.process()
    print(f"\nProcessed enrollment data for {len(processed_df)} institutions")

    # Validation
    unique_unitids = processed_df["UNITID"].nunique()
    total_rows = len(processed_df)
    print(f"Unique UNITIDs: {unique_unitids}")
    print(f"Total rows: {total_rows}")
    if unique_unitids != total_rows:
        print("❌ ERROR: Duplicate UNITIDs detected!")
    else:
        print("✅ SUCCESS: No duplicate UNITIDs")
