#!/usr/bin/env python3
"""
IPEDS Data Processing Runner

This script runs the complete IPEDS data processing pipeline.
Run this from your project root directory.

Usage:
    python scripts/run_processing.py [--processors processor1,processor2,...]

Examples:
    python scripts/run_processing.py  # Process all data
    python scripts/run_processing.py --processors institutional_directory,admissions  # Process specific datasets
"""

import argparse
import sys
import os
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from master_processor import MasterIPEDSProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process IPEDS data for university search application"
    )
    parser.add_argument(
        "--processors",
        type=str,
        help="Comma-separated list of processors to run (institutional_directory,admissions,enrollment,finance)",
        default=None,
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="raw_data",
        help="Path to raw IPEDS data files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="processed_data",
        help="Path to save processed data files",
    )
    parser.add_argument(
        "--quick-only",
        action="store_true",
        help="Only run quick analysis on existing processed data",
    )

    args = parser.parse_args()

    # Validate paths
    raw_data_path = Path(args.raw_data_path)
    if not raw_data_path.exists():
        print(f"Error: Raw data directory '{raw_data_path}' does not exist")
        print("Make sure you have downloaded IPEDS data files to this directory")
        sys.exit(1)

    # Check for required files
    required_files = ["hd2023.csv"]  # At minimum, need institutional directory
    missing_files = [f for f in required_files if not (raw_data_path / f).exists()]
    if missing_files:
        print(f"Error: Required files missing from {raw_data_path}:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)

    # Initialize processor
    processor = MasterIPEDSProcessor(
        raw_data_path=str(raw_data_path), processed_data_path=args.output_path
    )

    if args.quick_only:
        print("Running quick analysis on existing processed data...")
        analysis = processor.quick_analysis()
        if analysis:
            print_analysis_results(analysis)
        else:
            print("No processed data found. Run without --quick-only first.")
        return

    # Parse processors to run
    processors_to_run = None
    if args.processors:
        processors_to_run = [p.strip() for p in args.processors.split(",")]
        valid_processors = [
            "institutional_directory",
            "admissions",
            "enrollment",
            "finance",
        ]
        invalid_processors = [p for p in processors_to_run if p not in valid_processors]
        if invalid_processors:
            print(f"Error: Invalid processors: {invalid_processors}")
            print(f"Valid processors: {valid_processors}")
            sys.exit(1)

    print("🏫 IPEDS University Data Processing Pipeline")
    print("=" * 50)
    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {args.output_path}")

    if processors_to_run:
        print(f"Processing: {', '.join(processors_to_run)}")
    else:
        print("Processing: All datasets")

    print()

    try:
        # Run processing
        processed_data = processor.process_all(processors_to_run)

        # Create unified dataset
        unified_df = processor.create_unified_dataset(processed_data)

        # Run analysis
        analysis = processor.quick_analysis(unified_df)

        # Print results
        print_analysis_results(analysis)

        print("\n✅ Processing complete!")
        print(f"\nGenerated files in '{args.output_path}':")
        print("  📊 unified_ipeds_dataset.csv - Main dataset for your application")
        print("  📋 processing_summary_report.txt - Detailed processing report")
        print(
            "  📁 Individual processed datasets (institutional_directory_processed.csv, etc.)"
        )

        print(f"\n🚀 Ready for the next steps:")
        print("  1. Import data into PostgreSQL database")
        print("  2. Build FastAPI backend with search endpoints")
        print("  3. Create React frontend for university search")

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print(
            "  - Check that all required IPEDS CSV files are in the raw_data directory"
        )
        print("  - Ensure files are not corrupted and have proper headers")
        print("  - Check file encoding (should be UTF-8 or Latin-1)")
        sys.exit(1)


def print_analysis_results(analysis):
    """Print formatted analysis results."""
    print("\n📈 Data Processing Results")
    print("-" * 30)

    total = analysis.get("total_institutions", 0)
    print(f"Total institutions: {total:,}")

    if "by_control_type" in analysis:
        print("\nInstitution types:")
        for control_type, count in analysis["by_control_type"].items():
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {control_type}: {count:,} ({pct:.1f}%)")

    if "acceptance_rate_stats" in analysis:
        stats = analysis["acceptance_rate_stats"]
        print(f"\nAcceptance rates:")
        print(f"  Median: {stats['median']:.1f}%")
        print(f"  Average: {stats['mean']:.1f}% (±{stats['std']:.1f}%)")

    if "cost_stats" in analysis:
        stats = analysis["cost_stats"]
        print(f"\nTuition & fees (in-state):")
        print(f"  Median: ${stats['median']:,.0f}")
        print(f"  Average: ${stats['mean']:,.0f} (±${stats['std']:,.0f})")


if __name__ == "__main__":
    main()
