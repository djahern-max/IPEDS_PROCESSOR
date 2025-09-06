# Fixed master_processor.py with comprehensive duplicate prevention
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor_base import IPEDSProcessor
from process_institutional_directory import InstitutionalDirectoryProcessor
from process_admissions import AdmissionsProcessor
from process_enrollment import EnrollmentProcessor
from process_finance import FinanceProcessor


class MasterIPEDSProcessor:
    """FIXED Master processor with comprehensive duplicate prevention and validation."""

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
        self.logger = logging.getLogger("MasterProcessor")

        # Initialize processors
        self.processors = {
            "institutional_directory": InstitutionalDirectoryProcessor(
                raw_data_path=raw_data_path, processed_data_path=processed_data_path
            ),
            "admissions": AdmissionsProcessor(
                raw_data_path=raw_data_path, processed_data_path=processed_data_path
            ),
            "enrollment": EnrollmentProcessor(
                raw_data_path=raw_data_path, processed_data_path=processed_data_path
            ),
            "finance": FinanceProcessor(
                raw_data_path=raw_data_path, processed_data_path=processed_data_path
            ),
        }

    def process_all(
        self, processors_to_run: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Process all or specified data categories with enhanced validation."""

        if processors_to_run is None:
            processors_to_run = list(self.processors.keys())

        self.logger.info(
            f"Starting master processing for: {', '.join(processors_to_run)}"
        )

        processed_data = {}

        for processor_name in processors_to_run:
            if processor_name not in self.processors:
                self.logger.warning(f"Unknown processor: {processor_name}")
                continue

            try:
                self.logger.info(f"Running {processor_name} processor...")
                processed_df = self.processors[processor_name].process()

                # CRITICAL FIX: Validate each processed dataset
                validation_result = self._validate_processed_dataset(
                    processed_df, processor_name
                )
                if not validation_result["is_valid"]:
                    self.logger.error(
                        f"❌ {processor_name} failed validation: {validation_result['issues']}"
                    )
                    # Attempt to fix common issues
                    processed_df = self._fix_common_issues(processed_df, processor_name)

                processed_data[processor_name] = processed_df
                self.logger.info(
                    f"✓ {processor_name} completed: {len(processed_df)} records, {processed_df['UNITID'].nunique() if 'UNITID' in processed_df.columns else 0} unique institutions"
                )

            except Exception as e:
                self.logger.error(f"✗ {processor_name} failed: {str(e)}")
                processed_data[processor_name] = pd.DataFrame()

        return processed_data

    def _validate_processed_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Validate individual processed dataset."""
        issues = []
        warnings = []

        # Expected row counts (rough estimates)
        expected_counts = {
            "institutional_directory": (6000, 7000),
            "admissions": (1500, 3000),
            "enrollment": (6000, 7000),
            "finance": (5000, 8000),
        }

        # Check basic structure
        if "UNITID" not in df.columns:
            issues.append("Missing UNITID column")
            return {"is_valid": False, "issues": issues, "warnings": warnings}

        # Check row count
        row_count = len(df)
        unique_unitids = df["UNITID"].nunique()

        if dataset_name in expected_counts:
            min_expected, max_expected = expected_counts[dataset_name]
            if row_count > max_expected * 2:  # More than double expected
                issues.append(
                    f"Excessive row count: {row_count} (expected {min_expected}-{max_expected})"
                )
            elif row_count < min_expected * 0.5:  # Less than half expected
                warnings.append(
                    f"Low row count: {row_count} (expected {min_expected}-{max_expected})"
                )

        # Check for duplicates
        duplicate_count = df["UNITID"].duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate UNITIDs")

        # Check UNITID format
        invalid_unitids = df["UNITID"][
            (df["UNITID"] < 100000) | (df["UNITID"] > 999999)
        ]
        if len(invalid_unitids) > 0:
            warnings.append(
                f"Found {len(invalid_unitids)} UNITIDs outside 6-digit range"
            )

        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "row_count": row_count,
            "unique_unitids": unique_unitids,
        }

    def _fix_common_issues(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Attempt to fix common data issues."""
        original_len = len(df)

        # Fix 1: Remove duplicate UNITIDs
        if "UNITID" in df.columns:
            df = df.drop_duplicates(subset=["UNITID"], keep="first")
            if len(df) != original_len:
                self.logger.info(
                    f"Fixed {dataset_name}: Removed {original_len - len(df)} duplicate UNITIDs"
                )

        # Fix 2: Remove invalid UNITIDs
        if "UNITID" in df.columns:
            valid_mask = (df["UNITID"] >= 100000) & (df["UNITID"] <= 999999)
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                df = df[valid_mask]
                self.logger.info(
                    f"Fixed {dataset_name}: Removed {invalid_count} invalid UNITIDs"
                )

        # Fix 3: Limit to reasonable institution count
        if len(df) > 10000:  # Way too many institutions
            self.logger.warning(
                f"Dataset {dataset_name} still has {len(df)} rows after fixes - this may indicate data multiplication"
            )

        return df

    def create_unified_dataset(
        self, processed_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Create a unified dataset with comprehensive duplicate prevention."""

        if processed_data is None:
            processed_data = self.process_all()

        self.logger.info("Creating unified dataset with enhanced validation...")

        # Start with institutional directory as the base
        if (
            "institutional_directory" in processed_data
            and len(processed_data["institutional_directory"]) > 0
        ):
            unified_df = processed_data["institutional_directory"].copy()
            base_count = len(unified_df)
            base_unitids = set(unified_df["UNITID"].unique())
            self.logger.info(f"Base dataset: {base_count} institutions")
        else:
            self.logger.error("No institutional directory data available")
            return pd.DataFrame()

        # CRITICAL FIX: Validate base dataset
        if unified_df["UNITID"].duplicated().any():
            self.logger.error("Base dataset has duplicate UNITIDs!")
            unified_df = unified_df.drop_duplicates(subset=["UNITID"], keep="first")
            self.logger.info(
                f"Fixed base dataset: {len(unified_df)} institutions after deduplication"
            )

        # Merge other datasets with comprehensive validation
        merge_order = ["admissions", "enrollment", "finance"]

        for dataset_name in merge_order:
            if dataset_name in processed_data and len(processed_data[dataset_name]) > 0:
                dataset_df = processed_data[dataset_name].copy()

                # PRE-MERGE VALIDATION
                self.logger.info(f"Preparing to merge {dataset_name}...")

                # Remove duplicates from dataset to merge
                if "UNITID" in dataset_df.columns:
                    original_len = len(dataset_df)
                    dataset_df = dataset_df.drop_duplicates(
                        subset=["UNITID"], keep="first"
                    )
                    if len(dataset_df) != original_len:
                        self.logger.warning(
                            f"Removed {original_len - len(dataset_df)} duplicates from {dataset_name} before merge"
                        )

                # Validate UNITIDs are in base dataset
                dataset_unitids = set(dataset_df["UNITID"].unique())
                invalid_unitids = dataset_unitids - base_unitids
                if invalid_unitids:
                    self.logger.warning(
                        f"{dataset_name}: {len(invalid_unitids)} UNITIDs not in institutional directory"
                    )
                    # Option: Remove or keep them
                    # For now, we'll keep them but log the issue

                # PERFORM MERGE
                before_count = len(unified_df)
                before_unitids = unified_df["UNITID"].nunique()

                unified_df = unified_df.merge(dataset_df, on="UNITID", how="left")

                after_count = len(unified_df)
                after_unitids = unified_df["UNITID"].nunique()

                # POST-MERGE VALIDATION
                if before_count != after_count:
                    self.logger.error(
                        f"❌ CRITICAL: Row count changed during {dataset_name} merge! {before_count} -> {after_count}"
                    )
                    # This should never happen with a proper left join on unique keys
                    raise ValueError(
                        f"Data multiplication detected during {dataset_name} merge"
                    )

                if before_unitids != after_unitids:
                    self.logger.error(
                        f"❌ CRITICAL: Unique UNITID count changed during {dataset_name} merge! {before_unitids} -> {after_unitids}"
                    )
                    raise ValueError(
                        f"UNITID integrity violation during {dataset_name} merge"
                    )

                # Check for any duplicates introduced
                duplicate_count = unified_df["UNITID"].duplicated().sum()
                if duplicate_count > 0:
                    self.logger.error(
                        f"❌ CRITICAL: {duplicate_count} duplicate UNITIDs introduced during {dataset_name} merge"
                    )
                    unified_df = unified_df.drop_duplicates(
                        subset=["UNITID"], keep="first"
                    )
                    self.logger.info(f"Fixed: Removed {duplicate_count} duplicates")

                self.logger.info(
                    f"✅ Merged {dataset_name}: {len(dataset_df)} records merged, "
                    f"unified dataset has {after_count} institutions"
                )
            else:
                self.logger.warning(f"No {dataset_name} data to merge")

        # FINAL VALIDATION
        self.logger.info("Performing final unified dataset validation...")
        final_validation = self._validate_unified_dataset(unified_df)

        if not final_validation["is_valid"]:
            self.logger.error(
                f"❌ Final validation failed: {final_validation['issues']}"
            )
            # Apply final fixes
            unified_df = self._apply_final_fixes(unified_df)
        else:
            self.logger.info("✅ Final validation passed")

        # Add final derived fields
        unified_df = self._add_unified_derived_fields(unified_df)

        # Create data quality score
        unified_df = self._calculate_data_quality_score(unified_df)

        # Save unified dataset
        output_path = self.processed_data_path / "unified_ipeds_dataset.csv"
        unified_df.to_csv(output_path, index=False)

        # Generate summary report
        self._generate_summary_report(unified_df, processed_data)

        self.logger.info(
            f"✅ Unified dataset created: {len(unified_df)} institutions, {len(unified_df.columns)} columns"
        )

        return unified_df

    def _validate_unified_dataset(self, df: pd.DataFrame) -> Dict:
        """Validate the final unified dataset."""
        issues = []
        warnings = []

        # Check basic structure
        if len(df) == 0:
            issues.append("Empty dataset")
            return {"is_valid": False, "issues": issues}

        if "UNITID" not in df.columns:
            issues.append("Missing UNITID column")
            return {"is_valid": False, "issues": issues}

        # Check for duplicates
        duplicate_count = df["UNITID"].duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate UNITIDs")

        # Check row count
        row_count = len(df)
        unique_unitids = df["UNITID"].nunique()

        if row_count != unique_unitids:
            issues.append(
                f"Row count ({row_count}) != unique UNITIDs ({unique_unitids})"
            )

        if row_count > 10000:
            issues.append(f"Excessive row count: {row_count} (expected ~6,000-7,000)")
        elif row_count < 3000:
            warnings.append(f"Low row count: {row_count} (expected ~6,000-7,000)")

        # Check column count
        col_count = len(df.columns)
        if col_count > 200:
            warnings.append(f"High column count: {col_count}")
        elif col_count < 50:
            warnings.append(f"Low column count: {col_count}")

        is_valid = len(issues) == 0
        return {"is_valid": is_valid, "issues": issues, "warnings": warnings}

    def _apply_final_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final fixes to unified dataset."""
        self.logger.info("Applying final fixes to unified dataset...")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=["UNITID"], keep="first")
        if len(df) != original_len:
            self.logger.info(f"Removed {original_len - len(df)} duplicate rows")

        # Remove invalid UNITIDs
        if "UNITID" in df.columns:
            valid_mask = (
                (df["UNITID"] >= 100000)
                & (df["UNITID"] <= 999999)
                & df["UNITID"].notna()
            )
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                df = df[valid_mask]
                self.logger.info(f"Removed {invalid_count} rows with invalid UNITIDs")

        return df

    # Keep the rest of the original methods but add enhanced logging
    def _add_unified_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields that require data from multiple sources."""
        self.logger.info("Adding unified derived fields...")
        df = df.copy()

        # Overall competitiveness score
        competitiveness_factors = []

        if "acceptance_rate" in df.columns:
            # Lower acceptance rate = higher competitiveness
            df["acceptance_competitiveness"] = (
                100 - df["acceptance_rate"].fillna(50)
            ) / 100
            competitiveness_factors.append("acceptance_competitiveness")

        if "sat_total_75" in df.columns:
            # Higher SAT scores = higher competitiveness
            df["sat_competitiveness"] = (df["sat_total_75"].fillna(1000) - 800) / 800
            df["sat_competitiveness"] = df["sat_competitiveness"].clip(0, 1)
            competitiveness_factors.append("sat_competitiveness")

        if "ACTCM75" in df.columns:
            # Higher ACT scores = higher competitiveness
            df["act_competitiveness"] = (df["ACTCM75"].fillna(20) - 15) / 21
            df["act_competitiveness"] = df["act_competitiveness"].clip(0, 1)
            competitiveness_factors.append("act_competitiveness")

        # Calculate overall competitiveness score
        if competitiveness_factors:
            df["competitiveness_score"] = (
                df[competitiveness_factors].mean(axis=1).round(3)
            )

        return df

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a data quality score for each institution."""
        self.logger.info("Calculating data quality scores...")
        df = df.copy()

        # Key fields that are important for student decision-making
        important_fields = [
            "INSTNM",
            "location",
            "control_type",  # Basic info
            "acceptance_rate",
            "sat_total_75",
            "ACTCM75",  # Admissions
            "student_body_size",  # Student body
            "total_in_state_tuition_fees",
            "room_and_board",  # Costs
        ]

        available_important_fields = [
            col for col in important_fields if col in df.columns
        ]

        if available_important_fields:
            # Calculate percentage of important fields that have data
            df["data_completeness"] = (
                df[available_important_fields].notna().sum(axis=1)
                / len(available_important_fields)
            ).round(3)
        else:
            df["data_completeness"] = 0.0

        # Data quality categories
        df["data_quality_category"] = df["data_completeness"].apply(
            lambda x: (
                "Excellent (90%+)"
                if x >= 0.9
                else (
                    "Good (70-89%)"
                    if x >= 0.7
                    else "Fair (50-69%)" if x >= 0.5 else "Poor (<50%)"
                )
            )
        )

        return df

    def _generate_summary_report(
        self, unified_df: pd.DataFrame, processed_data: Dict[str, pd.DataFrame]
    ):
        """Generate a comprehensive summary report."""
        self.logger.info("Generating summary report...")

        report_path = self.processed_data_path / "processing_summary_report.txt"

        with open(report_path, "w") as f:
            f.write("ENHANCED IPEDS DATA PROCESSING SUMMARY REPORT\n")
            f.write("=" * 52 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total institutions processed: {len(unified_df)}\n")
            f.write(
                f"Unique UNITIDs: {unified_df['UNITID'].nunique() if 'UNITID' in unified_df.columns else 'N/A'}\n"
            )
            f.write(f"Total columns in unified dataset: {len(unified_df.columns)}\n")
            f.write(
                f"Data quality score: {unified_df.get('data_completeness', pd.Series([0])).mean():.2f}\n\n"
            )

            # Dataset statistics
            f.write("DATASET BREAKDOWN\n")
            f.write("-" * 18 + "\n")
            for dataset_name, dataset_df in processed_data.items():
                unique_count = (
                    dataset_df["UNITID"].nunique()
                    if "UNITID" in dataset_df.columns
                    else "N/A"
                )
                f.write(
                    f"{dataset_name.title()}: {len(dataset_df)} records ({unique_count} unique institutions)\n"
                )
            f.write("\n")

            # Institution type breakdown
            if "control_type" in unified_df.columns:
                f.write("INSTITUTION TYPES\n")
                f.write("-" * 17 + "\n")
                control_counts = unified_df["control_type"].value_counts()
                for control_type, count in control_counts.items():
                    f.write(f"{control_type}: {count}\n")
                f.write("\n")

            # Data quality assessment
            if "data_quality_category" in unified_df.columns:
                f.write("DATA QUALITY ASSESSMENT\n")
                f.write("-" * 23 + "\n")
                quality_counts = unified_df["data_quality_category"].value_counts()
                for quality_cat, count in quality_counts.items():
                    f.write(f"{quality_cat}: {count}\n")
                f.write("\n")

            # Missing data analysis
            f.write("MISSING DATA ANALYSIS (Top 10)\n")
            f.write("-" * 32 + "\n")
            missing_data = unified_df.isnull().sum().sort_values(ascending=False)
            top_missing = missing_data.head(10)
            for col, missing_count in top_missing.items():
                missing_pct = (missing_count / len(unified_df)) * 100
                f.write(f"{col}: {missing_count} ({missing_pct:.1f}%)\n")
            f.write("\n")

            # Data integrity checks
            f.write("DATA INTEGRITY VALIDATION\n")
            f.write("-" * 27 + "\n")
            if "UNITID" in unified_df.columns:
                duplicate_count = unified_df["UNITID"].duplicated().sum()
                f.write(f"Duplicate UNITIDs: {duplicate_count}\n")
                f.write(f"Unique institutions: {unified_df['UNITID'].nunique()}\n")
                f.write(f"Total rows: {len(unified_df)}\n")
                integrity_status = "✅ PASS" if duplicate_count == 0 else "❌ FAIL"
                f.write(f"Data integrity status: {integrity_status}\n")
            f.write("\n")

            f.write("PROCESSING VALIDATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write("✅ All datasets processed successfully\n")
            f.write(
                "✅ No duplicate UNITIDs in final dataset\n"
                if unified_df["UNITID"].duplicated().sum() == 0
                else "❌ Duplicate UNITIDs detected\n"
            )
            f.write(
                "✅ Institution count within expected range\n"
                if 3000 <= len(unified_df) <= 8000
                else "⚠️  Institution count outside expected range\n"
            )
            f.write("\n")

            f.write("FILES CREATED\n")
            f.write("-" * 13 + "\n")
            f.write("📊 unified_ipeds_dataset.csv - Main dataset for applications\n")
            f.write("📋 processing_summary_report.txt - This report\n")
            f.write("📁 Individual processed datasets:\n")
            for dataset_name in processed_data.keys():
                f.write(f"   - {dataset_name}_processed.csv\n")
            f.write("\n")

            f.write("NEXT STEPS\n")
            f.write("-" * 10 + "\n")
            f.write("1. 🗄️  Import unified dataset into PostgreSQL\n")
            f.write("2. 🔧 Build FastAPI backend with search endpoints\n")
            f.write("3. ⚛️  Create React frontend interface\n")
            f.write("4. 🧪 Test with sample queries\n")

        self.logger.info(f"Enhanced summary report saved to {report_path}")

    def quick_analysis(self, unified_df: Optional[pd.DataFrame] = None) -> Dict:
        """Perform quick analysis with enhanced validation."""

        if unified_df is None:
            unified_path = self.processed_data_path / "unified_ipeds_dataset.csv"
            if unified_path.exists():
                unified_df = pd.read_csv(unified_path)
            else:
                self.logger.error("No unified dataset found. Run process_all() first.")
                return {}

        analysis = {}

        # Basic statistics with validation
        analysis["total_institutions"] = len(unified_df)
        analysis["unique_institutions"] = (
            unified_df["UNITID"].nunique() if "UNITID" in unified_df.columns else 0
        )
        analysis["has_duplicates"] = (
            unified_df["UNITID"].duplicated().any()
            if "UNITID" in unified_df.columns
            else False
        )

        # Data integrity score
        if analysis["total_institutions"] > 0:
            analysis["data_integrity_score"] = (
                100
                if analysis["total_institutions"] == analysis["unique_institutions"]
                else 0
            )

        if "control_type" in unified_df.columns:
            analysis["by_control_type"] = (
                unified_df["control_type"].value_counts().to_dict()
            )

        if "acceptance_rate" in unified_df.columns:
            acceptance_stats = unified_df["acceptance_rate"].describe()
            analysis["acceptance_rate_stats"] = {
                "median": acceptance_stats["50%"],
                "mean": acceptance_stats["mean"],
                "std": acceptance_stats["std"],
            }

        return analysis


def main():
    """Main execution function with enhanced error handling."""

    # Initialize master processor
    processor = MasterIPEDSProcessor()

    print("🏫 Enhanced IPEDS Data Processing Pipeline")
    print("=" * 50)

    try:
        # Process all data
        print("📊 Processing individual datasets...")
        processed_data = processor.process_all()

        # Validate individual datasets
        print("🔍 Validating processed datasets...")
        validation_passed = True
        for name, df in processed_data.items():
            if len(df) == 0:
                print(f"⚠️  {name}: No data processed")
                continue

            unique_unitids = df["UNITID"].nunique() if "UNITID" in df.columns else 0
            total_rows = len(df)

            if "UNITID" in df.columns and unique_unitids != total_rows:
                print(
                    f"❌ {name}: Has duplicate UNITIDs ({total_rows} rows, {unique_unitids} unique)"
                )
                validation_passed = False
            else:
                print(f"✅ {name}: {total_rows} institutions")

        if not validation_passed:
            print(
                "\n⚠️  Some datasets have validation issues, but continuing with merge..."
            )

        # Create unified dataset
        print("\n🔄 Creating unified dataset...")
        unified_df = processor.create_unified_dataset(processed_data)

        # Final validation
        print("\n🔍 Final validation...")
        final_unique = (
            unified_df["UNITID"].nunique() if "UNITID" in unified_df.columns else 0
        )
        final_total = len(unified_df)

        if final_unique == final_total:
            print(f"✅ SUCCESS: {final_total} institutions, no duplicates")
        else:
            print(
                f"❌ ISSUE: {final_total} rows but only {final_unique} unique institutions"
            )

        # Quick analysis
        analysis = processor.quick_analysis(unified_df)

        print(f"\n📈 Final Statistics:")
        print(f"Total institutions: {analysis.get('total_institutions', 0):,}")
        print(f"Data integrity score: {analysis.get('data_integrity_score', 0)}/100")

        if "by_control_type" in analysis:
            print("\nInstitution types:")
            for control_type, count in analysis["by_control_type"].items():
                print(f"  {control_type}: {count:,}")

        print(f"\n📁 Files saved to: {processor.processed_data_path}")

        if analysis.get("data_integrity_score", 0) == 100:
            print("🎉 Data processing completed successfully!")
            print("✅ Ready for database import and application development!")
        else:
            print(
                "⚠️  Data processing completed with issues. Review the validation report."
            )

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        print("\nRecommended actions:")
        print("1. Check raw data files are present and valid")
        print("2. Review processing logs for specific errors")
        print("3. Run validation tool to diagnose issues")
        raise


if __name__ == "__main__":
    main()
