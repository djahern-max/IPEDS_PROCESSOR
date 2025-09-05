# scripts/master_processor.py
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
    """Master processor that coordinates all IPEDS data processing."""

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
        """Process all or specified data categories."""

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
                processed_data[processor_name] = processed_df
                self.logger.info(
                    f"✓ {processor_name} completed: {len(processed_df)} records"
                )

            except Exception as e:
                self.logger.error(f"✗ {processor_name} failed: {str(e)}")
                processed_data[processor_name] = pd.DataFrame()

        return processed_data

    def create_unified_dataset(
        self, processed_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Create a unified dataset by merging all processed data."""

        if processed_data is None:
            processed_data = self.process_all()

        self.logger.info("Creating unified dataset...")

        # Start with institutional directory as the base
        if (
            "institutional_directory" in processed_data
            and len(processed_data["institutional_directory"]) > 0
        ):
            unified_df = processed_data["institutional_directory"].copy()
            base_count = len(unified_df)
            self.logger.info(f"Base dataset: {base_count} institutions")
        else:
            self.logger.error("No institutional directory data available")
            return pd.DataFrame()

        # Merge other datasets
        merge_order = ["admissions", "enrollment", "finance"]

        for dataset_name in merge_order:
            if dataset_name in processed_data and len(processed_data[dataset_name]) > 0:
                dataset_df = processed_data[dataset_name]

                # Merge on UNITID
                before_count = len(unified_df)
                unified_df = unified_df.merge(dataset_df, on="UNITID", how="left")
                after_count = len(unified_df)

                self.logger.info(
                    f"Merged {dataset_name}: {len(dataset_df)} records, "
                    f"unified dataset still has {after_count} institutions"
                )

                if before_count != after_count:
                    self.logger.warning(
                        f"Row count changed during {dataset_name} merge!"
                    )
            else:
                self.logger.warning(f"No {dataset_name} data to merge")

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
            f"✓ Unified dataset created: {len(unified_df)} institutions, {len(unified_df.columns)} columns"
        )

        return unified_df

    def _add_unified_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields that require data from multiple sources."""
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

        # Value score (quality vs cost)
        value_factors = []

        if "competitiveness_score" in df.columns:
            value_factors.append("competitiveness_score")

        if "total_in_state_tuition_fees" in df.columns:
            # Lower cost = higher value (inverse relationship)
            max_cost = df["total_in_state_tuition_fees"].max()
            if max_cost and max_cost > 0:
                df["cost_value"] = 1 - (
                    df["total_in_state_tuition_fees"].fillna(max_cost) / max_cost
                )
                value_factors.append("cost_value")

        if value_factors and len(value_factors) >= 2:
            df["value_score"] = df[value_factors].mean(axis=1).round(3)

        # Student experience indicators
        experience_factors = []

        if "student_body_size" in df.columns:
            # Medium-sized institutions might offer better experience (inverted U-curve)
            df["size_experience"] = df["student_body_size"].apply(
                self._calculate_size_experience_score
            )
            experience_factors.append("size_experience")

        if "diversity_index" in df.columns:
            experience_factors.append("diversity_index")

        if experience_factors:
            df["student_experience_score"] = (
                df[experience_factors].mean(axis=1).round(3)
            )

        # Create recommendation flags
        if "competitiveness_score" in df.columns and "value_score" in df.columns:
            df["hidden_gem"] = (
                (df["competitiveness_score"] >= 0.6) & (df["value_score"] >= 0.7)
            ).astype(int)

        if "acceptance_rate" in df.columns and "student_experience_score" in df.columns:
            df["good_for_most_students"] = (
                (df["acceptance_rate"] >= 30)
                & (df["acceptance_rate"] <= 70)
                & (df["student_experience_score"] >= 0.6)
            ).astype(int)

        # Safety/match/reach categorization helper
        if "acceptance_rate" in df.columns:

            def categorize_admission_difficulty(rate):
                if pd.isna(rate):
                    return "Unknown"
                elif rate >= 75:
                    return "Safety"
                elif rate >= 50:
                    return "Match"
                elif rate >= 25:
                    return "Reach"
                else:
                    return "High Reach"

            df["admission_difficulty"] = df["acceptance_rate"].apply(
                categorize_admission_difficulty
            )

        return df

    def _calculate_size_experience_score(self, size):
        """Calculate experience score based on institution size (inverted U-curve)."""
        if pd.isna(size) or size <= 0:
            return 0.5

        # Optimal size around 5,000-15,000 students
        if 5000 <= size <= 15000:
            return 1.0
        elif 2000 <= size < 5000 or 15000 < size <= 25000:
            return 0.8
        elif 1000 <= size < 2000 or 25000 < size <= 40000:
            return 0.6
        else:
            return 0.4

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a data quality score for each institution."""
        df = df.copy()

        # Key fields that are important for student decision-making
        important_fields = [
            "INSTNM",
            "location",
            "control_type",  # Basic info
            "acceptance_rate",
            "sat_total_75",
            "ACTCM75",  # Admissions
            "student_body_size",
            "diversity_index",  # Student body
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

        report_path = self.processed_data_path / "processing_summary_report.txt"

        with open(report_path, "w") as f:
            f.write("IPEDS DATA PROCESSING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total institutions processed: {len(unified_df)}\n")
            f.write(f"Total columns in unified dataset: {len(unified_df.columns)}\n\n")

            # Dataset statistics
            f.write("DATASET BREAKDOWN\n")
            f.write("-" * 18 + "\n")
            for dataset_name, dataset_df in processed_data.items():
                f.write(f"{dataset_name.title()}: {len(dataset_df)} institutions\n")
            f.write("\n")

            # Institution type breakdown
            if "control_type" in unified_df.columns:
                f.write("INSTITUTION TYPES\n")
                f.write("-" * 17 + "\n")
                control_counts = unified_df["control_type"].value_counts()
                for control_type, count in control_counts.items():
                    f.write(f"{control_type}: {count}\n")
                f.write("\n")

            # Size distribution
            if "enrollment_size_category" in unified_df.columns:
                f.write("SIZE DISTRIBUTION\n")
                f.write("-" * 17 + "\n")
                size_counts = unified_df["enrollment_size_category"].value_counts()
                for size_cat, count in size_counts.items():
                    f.write(f"{size_cat}: {count}\n")
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
            f.write("MISSING DATA ANALYSIS\n")
            f.write("-" * 21 + "\n")
            missing_data = unified_df.isnull().sum().sort_values(ascending=False)
            top_missing = missing_data.head(10)
            for col, missing_count in top_missing.items():
                missing_pct = (missing_count / len(unified_df)) * 100
                f.write(f"{col}: {missing_count} ({missing_pct:.1f}%)\n")
            f.write("\n")

            # Recommendations for students
            if "good_for_most_students" in unified_df.columns:
                good_for_most = unified_df["good_for_most_students"].sum()
                f.write("STUDENT RECOMMENDATIONS\n")
                f.write("-" * 22 + "\n")
                f.write(f"Institutions good for most students: {good_for_most}\n")

            if "hidden_gem" in unified_df.columns:
                hidden_gems = unified_df["hidden_gem"].sum()
                f.write(f"Hidden gem institutions: {hidden_gems}\n")

            f.write("\n")
            f.write("Report generated successfully!\n")
            f.write("Files created:\n")
            f.write("- unified_ipeds_dataset.csv (main dataset)\n")
            f.write("- institutional_directory_processed.csv\n")
            f.write("- admissions_processed.csv\n")
            f.write("- enrollment_processed.csv\n")
            f.write("- finance_processed.csv\n")

        self.logger.info(f"Summary report saved to {report_path}")

    def quick_analysis(self, unified_df: Optional[pd.DataFrame] = None) -> Dict:
        """Perform quick analysis of the processed data."""

        if unified_df is None:
            unified_path = self.processed_data_path / "unified_ipeds_dataset.csv"
            if unified_path.exists():
                unified_df = pd.read_csv(unified_path)
            else:
                self.logger.error("No unified dataset found. Run process_all() first.")
                return {}

        analysis = {}

        # Basic statistics
        analysis["total_institutions"] = len(unified_df)

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

        if "total_in_state_tuition_fees" in unified_df.columns:
            cost_stats = unified_df["total_in_state_tuition_fees"].describe()
            analysis["cost_stats"] = {
                "median": cost_stats["50%"],
                "mean": cost_stats["mean"],
                "std": cost_stats["std"],
            }

        return analysis


def main():
    """Main execution function."""

    # Initialize master processor
    processor = MasterIPEDSProcessor()

    print("Starting IPEDS data processing pipeline...")
    print("=" * 50)

    # Process all data
    processed_data = processor.process_all()

    # Create unified dataset
    unified_df = processor.create_unified_dataset(processed_data)

    # Quick analysis
    analysis = processor.quick_analysis(unified_df)

    print("\nProcessing complete!")
    print(f"✓ Processed {analysis.get('total_institutions', 0)} institutions")
    print("\nKey statistics:")

    if "by_control_type" in analysis:
        print("Institution types:")
        for control_type, count in analysis["by_control_type"].items():
            print(f"  {control_type}: {count}")

    if "acceptance_rate_stats" in analysis:
        print(f"\nAcceptance rates:")
        print(f"  Median: {analysis['acceptance_rate_stats']['median']:.1f}%")
        print(f"  Average: {analysis['acceptance_rate_stats']['mean']:.1f}%")

    if "cost_stats" in analysis:
        print(f"\nTuition & Fees (in-state):")
        print(f"  Median: ${analysis['cost_stats']['median']:,.0f}")
        print(f"  Average: ${analysis['cost_stats']['mean']:,.0f}")

    print(f"\nFiles saved to: {processor.processed_data_path}")
    print("✓ Ready for database import and API development!")


if __name__ == "__main__":
    main()
