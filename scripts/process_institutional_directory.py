# scripts/process_institutional_directory.py
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class InstitutionalDirectoryProcessor(IPEDSProcessor):
    """Process HD2023 - Institutional Directory Information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Control type mappings
        self.control_mapping = {
            1: "Public",
            2: "Private nonprofit",
            3: "Private for-profit",
        }

        # Institutional level mappings
        self.level_mapping = {
            1: "Four or more years",
            2: "At least 2 but less than 4 years",
            3: "Less than 2 years",
            -1: "Not applicable",
            -2: "Not applicable",
        }

        # Carnegie Classification mappings (simplified)
        self.carnegie_mapping = {
            15: "Doctoral Universities: Very High Research Activity",
            16: "Doctoral Universities: High Research Activity",
            17: "Doctoral/Professional Universities",
            18: "Master's Colleges & Universities: Larger Programs",
            19: "Master's Colleges & Universities: Medium Programs",
            20: "Master's Colleges & Universities: Small Programs",
            21: "Baccalaureate Colleges: Arts & Sciences Focus",
            22: "Baccalaureate Colleges: Diverse Fields",
            23: "Baccalaureate/Associate's Colleges",
            24: "Associate's Colleges: High Transfer-High Traditional",
            25: "Associate's Colleges: High Transfer-Mixed Traditional/Nontraditional",
            26: "Associate's Colleges: High Transfer-High Nontraditional",
            27: "Associate's Colleges: Mixed Transfer/Career & Technical-High Traditional",
            28: "Associate's Colleges: Mixed Transfer/Career & Technical-Mixed Traditional/Nontraditional",
            29: "Associate's Colleges: Mixed Transfer/Career & Technical-High Nontraditional",
            30: "Associate's Colleges: High Career & Technical-High Traditional",
            31: "Associate's Colleges: High Career & Technical-Mixed Traditional/Nontraditional",
            32: "Associate's Colleges: High Career & Technical-High Nontraditional",
            33: "Special Focus Two-Year: Health Professions & Other Fields",
            34: "Special Focus Two-Year: Technical Professions",
            35: "Special Focus Four-Year: Faith-Related Institutions",
            36: "Special Focus Four-Year: Medical Schools & Medical Centers",
            37: "Special Focus Four-Year: Other Health Professions Schools",
            38: "Special Focus Four-Year: Engineering Schools",
            39: "Special Focus Four-Year: Other Technology-Related Schools",
            40: "Special Focus Four-Year: Business & Management Schools",
            41: "Special Focus Four-Year: Arts, Music & Design Schools",
            42: "Special Focus Four-Year: Law Schools",
            43: "Special Focus Four-Year: Other Special Focus Institutions",
            -1: "Not classified",
            -2: "Not classified",
        }

    def process(self) -> pd.DataFrame:
        """Process institutional directory data."""
        self.logger.info("Starting institutional directory processing...")

        # Load raw data
        df = self.load_csv("hd2023.csv")

        # Select key columns for student search
        key_columns = [
            "UNITID",
            "INSTNM",
            "IALIAS",  # Identifiers and names
            "ADDR",
            "CITY",
            "STABBR",
            "ZIP",
            "FIPS",  # Location
            "CHFNM",
            "CHFTITLE",  # Leadership
            "GENTELE",
            "WEBADDR",  # Contact
            "CONTROL",
            "ICLEVEL",
            "HLOFFER",  # Institution type
            "UGOFFER",
            "GROFFER",
            "HDEGOFR1",  # Degree offerings
            "DEGGRANT",
            "HBCU",
            "PBI",
            "ANNHI",
            "TRIBAL",  # Special designations
            "LANDGRNT",
            "INSTSIZE",
            "F1SYSTYP",  # Additional characteristics
            "CCBASIC",
            "CCIPUG",
            "CCIPGRAD",
            "CCUGPROF",
            "CCENRPRF",  # Carnegie classifications
            "CCSIZSET",
            "CARNEGIE",
            "TENURESYSTEM",  # More classifications
            "MEDICAL",
            "HOSPITAL",
            "CYACTIVE",  # Special programs/status
        ]

        # Filter to available columns
        available_columns = [col for col in key_columns if col in df.columns]
        df = df[available_columns].copy()

        # Clean text columns
        text_columns = [
            "INSTNM",
            "IALIAS",
            "ADDR",
            "CITY",
            "STABBR",
            "ZIP",
            "CHFNM",
            "CHFTITLE",
            "GENTELE",
            "WEBADDR",
        ]
        df = self.clean_text_columns(df, text_columns)

        # Clean numeric columns
        numeric_columns = [
            "UNITID",
            "FIPS",
            "CONTROL",
            "ICLEVEL",
            "HLOFFER",
            "UGOFFER",
            "GROFFER",
            "HDEGOFR1",
            "DEGGRANT",
            "HBCU",
            "PBI",
            "ANNHI",
            "TRIBAL",
            "LANDGRNT",
            "INSTSIZE",
            "CCBASIC",
            "MEDICAL",
            "HOSPITAL",
            "CYACTIVE",
        ]
        df = self.clean_numeric_columns(df, numeric_columns)

        # Add derived fields
        df = self.add_derived_fields(df)

        # Validate and save
        validation_info = self.validate_data(df)
        self.save_processed_data(
            df, "institutional_directory_processed.csv", validation_info
        )

        return df

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields specific to institutional directory."""
        df = df.copy()

        # Add human-readable control type
        if "CONTROL" in df.columns:
            df["control_type"] = df["CONTROL"].map(self.control_mapping)

        # Add human-readable level
        if "ICLEVEL" in df.columns:
            df["institutional_level"] = df["ICLEVEL"].map(self.level_mapping)

        # Add Carnegie classification description
        if "CCBASIC" in df.columns:
            df["carnegie_basic_desc"] = df["CCBASIC"].map(self.carnegie_mapping)

        # Create size categories
        if "INSTSIZE" in df.columns:
            size_mapping = {
                1: "Very small (under 1,000)",
                2: "Small (1,000-2,999)",
                3: "Medium (3,000-9,999)",
                4: "Large (10,000-19,999)",
                5: "Very large (20,000 and above)",
                -1: "Not reported",
                -2: "Not applicable",
            }
            df["size_category"] = df["INSTSIZE"].map(size_mapping)

        # Create degree level flags
        if "HLOFFER" in df.columns:
            df["offers_graduate_degree"] = (df["HLOFFER"] >= 3).astype(int)

        # Create special designation flags
        minority_serving_cols = ["HBCU", "PBI", "ANNHI", "TRIBAL"]
        available_msi_cols = [col for col in minority_serving_cols if col in df.columns]
        if available_msi_cols:
            df["minority_serving_institution"] = df[available_msi_cols].sum(axis=1) > 0

        # Clean up web addresses
        if "WEBADDR" in df.columns:
            df["clean_website"] = df["WEBADDR"].str.replace(r"^www\.", "", regex=True)
            df["has_website"] = df["WEBADDR"].notna()

        # Create location string
        location_parts = []
        if "CITY" in df.columns:
            location_parts.append(df["CITY"])
        if "STABBR" in df.columns:
            location_parts.append(df["STABBR"])

        if location_parts:
            df["location"] = location_parts[0].str.cat(
                location_parts[1:], sep=", ", na_rep=""
            )

        return df


if __name__ == "__main__":
    processor = InstitutionalDirectoryProcessor()
    processed_df = processor.process()
    print(f"\nProcessed {len(processed_df)} institutions")
    print("\nSample processed data:")
    print(processed_df[["INSTNM", "location", "control_type", "size_category"]].head())
