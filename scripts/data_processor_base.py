# scripts/data_processor_base.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re


class IPEDSProcessor:
    """Base class for processing IPEDS data files with common utilities."""

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

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load CSV with proper encoding handling."""
        filepath = self.raw_data_path / filename

        # Try UTF-8 first, then latin-1
        for encoding in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                self.logger.info(
                    f"Loaded {filename} with {encoding} encoding: {len(df)} rows"
                )
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not load {filename} with any encoding")

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
                df[col] = pd.to_numeric(df[col], errors="coerce")

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
        """Validate processed data and return quality metrics."""
        validation = {
            "total_records": len(df),
            "duplicate_unitids": (
                df["UNITID"].duplicated().sum() if "UNITID" in df.columns else 0
            ),
            "missing_data_by_column": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }

        self.logger.info(
            f"Validation complete: {validation['total_records']} records processed"
        )
        return validation

    def save_processed_data(
        self, df: pd.DataFrame, filename: str, validation_info: Dict = None
    ):
        """Save processed data with validation report."""
        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)

        # Save validation report
        if validation_info:
            report_path = (
                self.processed_data_path
                / f"{filename.replace('.csv', '_validation.txt')}"
            )
            with open(report_path, "w") as f:
                f.write("Data Processing Validation Report\n")
                f.write("=" * 40 + "\n\n")
                for key, value in validation_info.items():
                    f.write(f"{key}: {value}\n")

        self.logger.info(f"Saved processed data to {output_path}")

    def process(self) -> pd.DataFrame:
        """Main processing method. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process() method")
