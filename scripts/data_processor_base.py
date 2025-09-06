# Enhanced data_processor_base.py with robust validation
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re


class IPEDSProcessor:
    """Enhanced base class for processing IPEDS data files with comprehensive validation."""

    def __init__(
        self,
        raw_data_path: str = "raw_data",
        processed_data_path: str = "processed_data",
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # IPEDS expected institution count (for validation)
        self.expected_max_institutions = (
            7000  # IPEDS has ~6,000-6,500 active institutions
        )

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load CSV with proper encoding handling and initial validation."""
        filepath = self.raw_data_path / filename

        # Try UTF-8 first, then latin-1
        for encoding in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                self.logger.info(
                    f"Loaded {filename} with {encoding} encoding: {len(df)} rows"
                )

                # ENHANCED: Immediate validation after load
                self._validate_raw_data(df, filename)

                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not load {filename} with any encoding")

    def _validate_raw_data(self, df: pd.DataFrame, filename: str):
        """Validate raw data immediately after loading."""
        # Check for UNITID column
        if "UNITID" not in df.columns:
            self.logger.warning(f"{filename}: No UNITID column found")
            return

        # Check UNITID format and range
        unitid_series = pd.to_numeric(df["UNITID"], errors="coerce")
        invalid_unitids = unitid_series[
            (unitid_series < 100000) | (unitid_series > 999999)
        ]
        if len(invalid_unitids) > 0:
            self.logger.warning(
                f"{filename}: Found {len(invalid_unitids)} UNITIDs outside 6-digit range"
            )

        # Check for duplicate UNITIDs in raw data
        duplicate_count = df["UNITID"].duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(
                f"{filename}: Found {duplicate_count} duplicate UNITIDs in raw data"
            )

        # Check row count sanity
        unique_unitids = df["UNITID"].nunique()
        if unique_unitids > self.expected_max_institutions:
            self.logger.error(
                f"{filename}: Too many unique UNITIDs ({unique_unitids}) - expected max {self.expected_max_institutions}"
            )

    def clean_numeric_columns(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Clean numeric columns by handling IPEDS null codes."""
        df = df.copy()

        # IPEDS uses specific codes for missing data
        null_codes = {
            ".": "Not applicable",
            "..": "Not available",
            "{": "Item not applicable",
            "†": "Not applicable",
            "‡": "Not available",
            "§": "Not available",
            "¶": "Not available",
        }

        for col in columns:
            if col in df.columns:
                # Replace null codes with NaN
                df[col] = df[col].replace(list(null_codes.keys()), np.nan)
                # Convert to numeric
                original_count = len(df)
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Log conversion issues
                null_count = df[col].isnull().sum()
                if null_count > original_count * 0.8:  # More than 80% null
                    self.logger.warning(
                        f"Column {col}: {null_count}/{original_count} ({null_count/original_count:.1%}) values are null after cleaning"
                    )

        return df

    def clean_text_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean text columns by standardizing formatting."""
        df = df.copy()

        for col in columns:
            if col in df.columns:
                # Strip whitespace and handle empty strings
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(["", "nan", "None"], np.nan)

        return df

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common derived fields. Override in subclasses."""
        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced validation of processed data with comprehensive quality metrics."""
        validation = {
            "total_records": len(df),
            "unique_unitids": df["UNITID"].nunique() if "UNITID" in df.columns else 0,
            "duplicate_unitids": (
                df["UNITID"].duplicated().sum() if "UNITID" in df.columns else 0
            ),
            "missing_data_by_column": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }

        # ENHANCED: Additional validation checks
        if "UNITID" in df.columns:
            # Check UNITID integrity
            unitid_series = df["UNITID"]
            validation.update(
                {
                    "unitid_min": unitid_series.min(),
                    "unitid_max": unitid_series.max(),
                    "unitid_null_count": unitid_series.isnull().sum(),
                }
            )

            # Data quality flags
            validation["data_quality_flags"] = {
                "has_duplicate_unitids": validation["duplicate_unitids"] > 0,
                "too_many_institutions": validation["unique_unitids"]
                > self.expected_max_institutions,
                "too_few_institutions": validation["unique_unitids"] < 1000,
                "has_invalid_unitids": (
                    ((unitid_series < 100000) | (unitid_series > 999999)).any()
                    if not unitid_series.isnull().all()
                    else False
                ),
            }

            # Overall data quality score
            quality_issues = sum(validation["data_quality_flags"].values())
            validation["data_quality_score"] = max(
                0, 100 - (quality_issues * 25)
            )  # Each issue -25 points

        # Log validation results
        self.logger.info(
            f"Validation complete: {validation['total_records']} records processed"
        )

        if validation.get("data_quality_flags"):
            issues = [k for k, v in validation["data_quality_flags"].items() if v]
            if issues:
                self.logger.warning(
                    f"Data quality issues detected: {', '.join(issues)}"
                )
            else:
                self.logger.info("✅ No data quality issues detected")

        return validation

    def save_processed_data(
        self, df: pd.DataFrame, filename: str, validation_info: Dict = None
    ):
        """Save processed data with enhanced validation report."""

        # ENHANCED: Final validation before saving
        if "UNITID" in df.columns:
            # Remove any remaining duplicates
            original_len = len(df)
            df = df.drop_duplicates(subset=["UNITID"], keep="first")
            if len(df) != original_len:
                self.logger.warning(
                    f"Removed {original_len - len(df)} duplicate UNITIDs before saving"
                )

        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)

        # Enhanced validation report
        if validation_info:
            report_path = (
                self.processed_data_path
                / f"{filename.replace('.csv', '_validation.txt')}"
            )
            with open(report_path, "w") as f:
                f.write("ENHANCED DATA PROCESSING VALIDATION REPORT\n")
                f.write("=" * 45 + "\n\n")

                # Basic stats
                f.write("BASIC STATISTICS\n")
                f.write("-" * 16 + "\n")
                for key, value in validation_info.items():
                    if key not in [
                        "missing_data_by_column",
                        "data_types",
                        "data_quality_flags",
                    ]:
                        f.write(f"{key}: {value}\n")
                f.write("\n")

                # Data quality assessment
                if "data_quality_flags" in validation_info:
                    f.write("DATA QUALITY ASSESSMENT\n")
                    f.write("-" * 23 + "\n")
                    flags = validation_info["data_quality_flags"]
                    for flag, has_issue in flags.items():
                        status = "❌ FAIL" if has_issue else "✅ PASS"
                        f.write(f"{flag}: {status}\n")

                    quality_score = validation_info.get("data_quality_score", 0)
                    f.write(f"\nOverall Quality Score: {quality_score}/100\n\n")

                # Missing data analysis
                f.write("MISSING DATA ANALYSIS\n")
                f.write("-" * 21 + "\n")
                missing_data = validation_info.get("missing_data_by_column", {})
                total_rows = validation_info.get("total_records", 1)

                for col, missing_count in sorted(
                    missing_data.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    missing_pct = (missing_count / total_rows) * 100
                    f.write(f"{col}: {missing_count} ({missing_pct:.1f}%)\n")

        self.logger.info(f"Saved processed data to {output_path}")

    def process(self) -> pd.DataFrame:
        """Main processing method. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process() method")
