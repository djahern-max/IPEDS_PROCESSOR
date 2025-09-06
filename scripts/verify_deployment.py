#!/usr/bin/env python3
"""
Deployment verification script for IPEDS processing fixes
"""

import sys
from pathlib import Path

def verify_files():
    """Verify all required files are present and updated."""
    
    print("🔍 Verifying deployment...")
    
    required_files = [
        'scripts/validate_ipeds_data.py',
        'scripts/data_processor_base.py',
        'scripts/process_enrollment.py', 
        'scripts/master_processor.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check for key fixes in files
    fixes_found = []
    
    # Check enrollment processor for deduplication
    with open('scripts/process_enrollment.py', 'r') as f:
        content = f.read()
        if 'drop_duplicates' in content and 'CRITICAL FIX' in content:
            fixes_found.append("✅ Enrollment processor has deduplication fixes")
        else:
            fixes_found.append("❌ Enrollment processor missing critical fixes")
    
    # Check master processor for validation
    with open('scripts/master_processor.py', 'r') as f:
        content = f.read()
        if '_validate_processed_dataset' in content and 'CRITICAL FIX' in content:
            fixes_found.append("✅ Master processor has validation fixes")
        else:
            fixes_found.append("❌ Master processor missing validation fixes")
    
    # Check base processor for enhanced validation
    with open('scripts/data_processor_base.py', 'r') as f:
        content = f.read()
        if 'expected_max_institutions' in content and 'ENHANCED' in content:
            fixes_found.append("✅ Base processor has enhanced validation")
        else:
            fixes_found.append("❌ Base processor missing enhanced validation")
    
    print("\n📋 Fix verification:")
    for fix in fixes_found:
        print(f"   {fix}")
    
    all_good = all("✅" in fix for fix in fixes_found)
    
    if all_good:
        print("\n🎉 All fixes deployed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/validate_ipeds_data.py")
        print("2. Run: python scripts/run_processing.py")
        print("3. Validate: python scripts/validate_ipeds_data.py")
        return True
    else:
        print("\n⚠️  Some fixes may not be properly deployed")
        return False

if __name__ == "__main__":
    success = verify_files()
    sys.exit(0 if success else 1)
