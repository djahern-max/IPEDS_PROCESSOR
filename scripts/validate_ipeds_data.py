#!/usr/bin/env python3
"""
IPEDS Data Validation Diagnostic Tool

This script diagnoses data quality issues in the processed IPEDS datasets.
Run this to identify and fix data duplication and validation problems.

Usage:
    python validate_ipeds_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IPEDSDataValidator:
    """Comprehensive IPEDS data validation and diagnostic tool."""
    
    def __init__(self, processed_data_path: str = "processed_data"):
        self.processed_data_path = Path(processed_data_path)
        self.issues = []
        
    def validate_all_datasets(self) -> Dict:
        """Run comprehensive validation on all processed datasets."""
        logger.info("Starting comprehensive IPEDS data validation...")
        
        validation_results = {}
        
        # Expected file structure
        expected_files = {
            'institutional_directory_processed.csv': 'Institutional Directory',
            'admissions_processed.csv': 'Admissions Data', 
            'enrollment_processed.csv': 'Enrollment Data',
            'finance_processed.csv': 'Finance Data',
            'unified_ipeds_dataset.csv': 'Unified Dataset'
        }
        
        # Validate each dataset
        for filename, description in expected_files.items():
            filepath = self.processed_data_path / filename
            
            if not filepath.exists():
                self.issues.append(f"Missing file: {filename}")
                validation_results[description] = {"status": "MISSING", "issues": [f"File not found: {filename}"]}
                continue
                
            try:
                logger.info(f"Validating {description}...")
                df = pd.read_csv(filepath, nrows=1000)  # Sample for quick validation
                full_info = self._get_file_info(filepath)
                validation_results[description] = self._validate_dataset(df, filename, full_info)
                
            except Exception as e:
                error_msg = f"Error reading {filename}: {str(e)}"
                self.issues.append(error_msg)
                validation_results[description] = {"status": "ERROR", "issues": [error_msg]}
        
        # Cross-dataset validation
        validation_results['Cross-Dataset Analysis'] = self._cross_validate_datasets()
        
        # Generate summary report
        self._generate_validation_report(validation_results)
        
        return validation_results
    
    def _get_file_info(self, filepath: Path) -> Dict:
        """Get basic file information without loading entire file."""
        try:
            # Get file size
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Get row count efficiently
            with open(filepath, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            
            # Get column count from header
            with open(filepath, 'r') as f:
                header = f.readline().strip()
                col_count = len(header.split(','))
            
            return {
                'file_size_mb': file_size_mb,
                'row_count': row_count,
                'col_count': col_count
            }
        except Exception as e:
            return {
                'file_size_mb': 0,
                'row_count': 0,
                'col_count': 0,
                'error': str(e)
            }
    
    def _validate_dataset(self, df: pd.DataFrame, filename: str, full_info: Dict) -> Dict:
        """Validate individual dataset."""
        issues = []
        warnings = []
        
        # Basic structural validation
        if 'UNITID' not in df.columns:
            issues.append("Missing UNITID column - required for all IPEDS datasets")
        
        # Check for duplicate UNITIDs in sample
        if 'UNITID' in df.columns:
            duplicate_count = df['UNITID'].duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Found {duplicate_count} duplicate UNITIDs in sample")
        
        # Validate expected row counts based on known IPEDS structure
        expected_ranges = {
            'institutional_directory_processed.csv': (6000, 7000),
            'admissions_processed.csv': (1500, 3000),
            'enrollment_processed.csv': (6000, 7000),
            'finance_processed.csv': (5000, 8000),
            'unified_ipeds_dataset.csv': (6000, 7000)
        }
        
        if filename in expected_ranges:
            expected_min, expected_max = expected_ranges[filename]
            actual_rows = full_info.get('row_count', 0)
            
            if actual_rows < expected_min:
                warnings.append(f"Row count ({actual_rows:,}) below expected minimum ({expected_min:,})")
            elif actual_rows > expected_max:
                issues.append(f"Row count ({actual_rows:,}) significantly exceeds expected maximum ({expected_max:,}) - possible data duplication")
        
        # File size validation
        file_size = full_info.get('file_size_mb', 0)
        if filename == 'unified_ipeds_dataset.csv' and file_size > 100:
            issues.append(f"Unified dataset file size ({file_size:.1f}MB) is unusually large - indicates data multiplication")
        elif filename == 'enrollment_processed.csv' and file_size > 50:
            issues.append(f"Enrollment file size ({file_size:.1f}MB) is unusually large - indicates data multiplication")
        
        # Data type validation
        if 'UNITID' in df.columns:
            if not pd.api.types.is_integer_dtype(df['UNITID']):
                issues.append("UNITID should be integer type")
            
            # Check UNITID format (should be 6-digit numbers)
            sample_unitids = df['UNITID'].dropna().head(10)
            if len(sample_unitids) > 0:
                invalid_unitids = sample_unitids[(sample_unitids < 100000) | (sample_unitids > 999999)]
                if len(invalid_unitids) > 0:
                    warnings.append("Found UNITIDs outside expected 6-digit range")
        
        # Missing data analysis
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 80]
        if len(high_missing) > 5:
            warnings.append(f"{len(high_missing)} columns have >80% missing data")
        
        # Determine status
        if issues:
            status = "CRITICAL_ISSUES"
        elif warnings:
            status = "WARNINGS" 
        else:
            status = "GOOD"
        
        return {
            'status': status,
            'file_info': full_info,
            'sample_shape': df.shape,
            'issues': issues,
            'warnings': warnings,
            'missing_data_summary': {
                'columns_over_50pct_missing': len(missing_pct[missing_pct > 50]),
                'columns_over_80pct_missing': len(missing_pct[missing_pct > 80]),
                'avg_missing_pct': missing_pct.mean()
            }
        }
    
    def _cross_validate_datasets(self) -> Dict:
        """Perform cross-dataset validation."""
        issues = []
        warnings = []
        
        # Try to load institutional directory as reference
        inst_dir_path = self.processed_data_path / 'institutional_directory_processed.csv'
        if inst_dir_path.exists():
            try:
                inst_df = pd.read_csv(inst_dir_path)
                reference_unitids = set(inst_df['UNITID'].unique())
                reference_count = len(reference_unitids)
                
                logger.info(f"Reference dataset has {reference_count} unique institutions")
                
                # Check other datasets against reference
                other_files = [
                    'admissions_processed.csv',
                    'enrollment_processed.csv', 
                    'finance_processed.csv'
                ]
                
                for filename in other_files:
                    filepath = self.processed_data_path / filename
                    if filepath.exists():
                        try:
                            # Sample the file to check UNITID distribution
                            sample_df = pd.read_csv(filepath, nrows=5000)
                            if 'UNITID' in sample_df.columns:
                                sample_unitids = set(sample_df['UNITID'].unique())
                                
                                # Check for UNITIDs not in reference
                                invalid_unitids = sample_unitids - reference_unitids
                                if invalid_unitids:
                                    issues.append(f"{filename}: Found {len(invalid_unitids)} UNITIDs not in institutional directory")
                                
                                # Check for excessive duplicate UNITIDs
                                duplicate_rate = sample_df['UNITID'].duplicated().mean()
                                if duplicate_rate > 0.5:
                                    issues.append(f"{filename}: {duplicate_rate:.1%} of rows are duplicate UNITIDs - data multiplication detected")
                                    
                        except Exception as e:
                            warnings.append(f"Could not validate {filename}: {str(e)}")
                            
            except Exception as e:
                issues.append(f"Could not load institutional directory for cross-validation: {str(e)}")
        else:
            issues.append("Institutional directory missing - cannot perform cross-validation")
        
        status = "CRITICAL_ISSUES" if issues else ("WARNINGS" if warnings else "GOOD")
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings
        }
    
    def _generate_validation_report(self, results: Dict):
        """Generate comprehensive validation report."""
        report_path = self.processed_data_path / 'data_validation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("IPEDS DATA VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall status
            critical_issues = sum(1 for r in results.values() if r.get('status') == 'CRITICAL_ISSUES')
            warning_datasets = sum(1 for r in results.values() if r.get('status') == 'WARNINGS')
            
            f.write("OVERALL STATUS\n")
            f.write("-" * 15 + "\n")
            if critical_issues > 0:
                f.write(f"🚨 CRITICAL: {critical_issues} datasets have critical issues\n")
            elif warning_datasets > 0:
                f.write(f"⚠️  WARNING: {warning_datasets} datasets have warnings\n")
            else:
                f.write("✅ GOOD: All datasets passed validation\n")
            f.write("\n")
            
            # Detailed results
            for dataset_name, result in results.items():
                f.write(f"{dataset_name.upper()}\n")
                f.write("-" * len(dataset_name) + "\n")
                f.write(f"Status: {result.get('status', 'UNKNOWN')}\n")
                
                if 'file_info' in result:
                    info = result['file_info']
                    f.write(f"File size: {info.get('file_size_mb', 0):.1f} MB\n")
                    f.write(f"Rows: {info.get('row_count', 0):,}\n")
                    f.write(f"Columns: {info.get('col_count', 0)}\n")
                
                if result.get('issues'):
                    f.write("ISSUES:\n")
                    for issue in result['issues']:
                        f.write(f"  • {issue}\n")
                
                if result.get('warnings'):
                    f.write("WARNINGS:\n")
                    for warning in result['warnings']:
                        f.write(f"  • {warning}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDED ACTIONS\n")
            f.write("-" * 19 + "\n")
            
            if critical_issues > 0:
                f.write("IMMEDIATE ACTIONS REQUIRED:\n")
                f.write("1. Check enrollment processing logic - likely data multiplication\n")
                f.write("2. Verify merge operations in master_processor.py\n")
                f.write("3. Re-run processing with fixed logic\n")
                f.write("4. Validate UNITID uniqueness in each dataset\n\n")
            
            f.write("GENERAL IMPROVEMENTS:\n")
            f.write("1. Add more robust duplicate detection\n")
            f.write("2. Implement row count validation checks\n")
            f.write("3. Add UNITID integrity constraints\n")
            f.write("4. Create automated validation pipeline\n")
        
        logger.info(f"Validation report saved to: {report_path}")

def main():
    """Run the validation diagnostic."""
    print("🔍 IPEDS Data Validation Diagnostic Tool")
    print("=" * 50)
    
    validator = IPEDSDataValidator()
    results = validator.validate_all_datasets()
    
    # Print summary
    critical_count = sum(1 for r in results.values() if r.get('status') == 'CRITICAL_ISSUES')
    warning_count = sum(1 for r in results.values() if r.get('status') == 'WARNINGS')
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("-" * 20)
    
    if critical_count > 0:
        print(f"🚨 {critical_count} datasets with CRITICAL ISSUES")
    elif warning_count > 0:
        print(f"⚠️  {warning_count} datasets with warnings")
    else:
        print("✅ All datasets passed validation!")
    
    print(f"\n📄 Detailed report saved to: processed_data/data_validation_report.txt")

if __name__ == "__main__":
    main()
