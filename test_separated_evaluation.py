#!/usr/bin/env python3
"""
Test script for the separated comprehensive evaluation functionality.
This script tests both evaluation-only and CSV-generation-only modes.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_evaluation_only():
    """Test evaluation-only mode with minimal parameters."""
    print("=== Testing Evaluation-Only Mode ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "comprehensive_evaluation_separated.py",
            "--mode", "evaluation",
            "--output", temp_dir,
            "--test",
            "--algorithms", "mmd",
            "--patterns", "default",
            "--seeds", "0"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                results_file = os.path.join(temp_dir, "evaluation_results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print(f"✓ Evaluation completed successfully with {len(results)} results")
                    return results_file, results
                else:
                    print("✗ Results file not found")
                    return None, None
            else:
                print(f"✗ Evaluation failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None, None
                
        except subprocess.TimeoutExpired:
            print("✗ Evaluation timed out")
            return None, None
        except Exception as e:
            print(f"✗ Evaluation failed with error: {e}")
            return None, None

def test_csv_generation(results_file, temp_dir):
    """Test CSV generation from existing results."""
    print("\n=== Testing CSV Generation-Only Mode ===")
    
    if not results_file or not os.path.exists(results_file):
        print("✗ No results file available for CSV generation test")
        return False
    
    cmd = [
        sys.executable, "comprehensive_evaluation_separated.py",
        "--mode", "csv",
        "--results-file", results_file,
        "--output", temp_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            expected_files = [
                "comprehensive_results.csv",
                "summary_statistics.csv",
                "confusion_matrices.csv",
                "confusion_matrix_tables.csv"
            ]
            
            missing_files = []
            for file in expected_files:
                if not os.path.exists(os.path.join(temp_dir, file)):
                    missing_files.append(file)
            
            if not missing_files:
                print("✓ CSV generation completed successfully")
                print(f"Generated files: {expected_files}")
                return True
            else:
                print(f"✗ Missing CSV files: {missing_files}")
                return False
        else:
            print(f"✗ CSV generation failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ CSV generation timed out")
        return False
    except Exception as e:
        print(f"✗ CSV generation failed with error: {e}")
        return False

def test_filtering():
    """Test filtering functionality."""
    print("\n=== Testing Filtering Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "comprehensive_evaluation_separated.py",
            "--mode", "evaluation",
            "--output", temp_dir,
            "--test",
            "--algorithms", "mmd", "lsdd",
            "--patterns", "default", "pattern1",
            "--scenarios", "datashift_on",
            "--seeds", "0"
        ]
        
        print(f"Running filtered evaluation: {' '.join(cmd)}")
        
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                results_file = os.path.join(temp_dir, "evaluation_results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    algorithms = set(r['algorithm'] for r in results)
                    patterns = set(r['pattern'] for r in results)
                    scenarios = set(r['scenario'] for r in results)
                    seeds = set(r['seed'] for r in results)
                    
                    expected_algorithms = {'mmd', 'lsdd'}
                    expected_patterns = {'default', 'pattern1'}
                    expected_scenarios = {'datashift_on'}
                    expected_seeds = {0}
                    
                    if (algorithms == expected_algorithms and 
                        patterns == expected_patterns and 
                        scenarios == expected_scenarios and 
                        seeds == expected_seeds):
                        print("✓ Filtering works correctly")
                        return True
                    else:
                        print(f"✗ Filtering failed:")
                        print(f"  Expected algorithms: {expected_algorithms}, got: {algorithms}")
                        print(f"  Expected patterns: {expected_patterns}, got: {patterns}")
                        print(f"  Expected scenarios: {expected_scenarios}, got: {scenarios}")
                        print(f"  Expected seeds: {expected_seeds}, got: {seeds}")
                        return False
                else:
                    print("✗ Results file not found")
                    return False
            else:
                print(f"✗ Filtered evaluation failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"✗ Filtering test failed with error: {e}")
            return False

def main():
    """Run all tests."""
    print("Starting separated evaluation functionality tests...")
    
    if not os.path.exists("comprehensive_evaluation_separated.py"):
        print("✗ comprehensive_evaluation_separated.py not found")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results_file, results = test_evaluation_only()
        
        if results_file and results:
            csv_success = test_csv_generation(results_file, temp_dir)
        else:
            csv_success = False
        
        filtering_success = test_filtering()
        
        print("\n=== Test Summary ===")
        print(f"Evaluation-only mode: {'✓ PASS' if results_file else '✗ FAIL'}")
        print(f"CSV generation mode: {'✓ PASS' if csv_success else '✗ FAIL'}")
        print(f"Filtering functionality: {'✓ PASS' if filtering_success else '✗ FAIL'}")
        
        all_passed = results_file and csv_success and filtering_success
        print(f"\nOverall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
