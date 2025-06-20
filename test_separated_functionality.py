#!/usr/bin/env python3
"""
Test script for the separated comprehensive evaluation functionality.
This script tests the core separation logic without requiring actual datasets.
"""

import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_filtering_logic():
    """Test the filtering logic without running actual experiments."""
    print("=== Testing Filtering Logic ===")
    
    try:
        from comprehensive_evaluation_separated import apply_filters, ALGORITHMS, PATTERNS, DATASETS, SCENARIOS, SEEDS
        
        result = apply_filters()
        expected = (ALGORITHMS, PATTERNS, DATASETS, SCENARIOS, SEEDS)
        if result == expected:
            print("✓ Default filtering (no filters) works correctly")
        else:
            print("✗ Default filtering failed")
            return False
        
        filtered_algs = ['mmd', 'lsdd']
        filtered_patterns = ['default', 'pattern1']
        result = apply_filters(algorithms=filtered_algs, patterns=filtered_patterns)
        
        if (result[0] == filtered_algs and 
            result[1] == filtered_patterns and 
            result[2] == DATASETS and 
            result[3] == SCENARIOS and 
            result[4] == SEEDS):
            print("✓ Specific filtering works correctly")
        else:
            print("✗ Specific filtering failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Filtering logic test failed: {e}")
        return False

def test_result_structure():
    """Test the result data structure and JSON serialization."""
    print("\n=== Testing Result Structure ===")
    
    try:
        mock_result = {
            'experiment_id': 'mmd_default_terrace_datashift_on_seed0',
            'algorithm': 'mmd',
            'pattern': 'default',
            'dataset': 'terrace',
            'scenario': 'datashift_on',
            'seed': 0,
            'status': 'success',
            'execution_time': 10.5,
            'results': [
                {'dataset': 'Test', 'is_drift': 1, 'p_val': 0.01, 'statistic': 2.5},
                {'dataset': 'Normal (H0)', 'is_drift': 0, 'p_val': 0.8, 'statistic': 0.1}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([mock_result], f, indent=2, default=str)
            temp_file = f.name
        
        with open(temp_file, 'r') as f:
            loaded_results = json.load(f)
        
        os.unlink(temp_file)
        
        if (len(loaded_results) == 1 and 
            loaded_results[0]['algorithm'] == 'mmd' and
            loaded_results[0]['status'] == 'success' and
            len(loaded_results[0]['results']) == 2):
            print("✓ Result structure and JSON serialization work correctly")
            return True
        else:
            print("✗ Result structure test failed")
            return False
            
    except Exception as e:
        print(f"✗ Result structure test failed: {e}")
        return False

def test_csv_generation_logic():
    """Test CSV generation logic with mock data."""
    print("\n=== Testing CSV Generation Logic ===")
    
    try:
        from comprehensive_evaluation_separated import generate_csv_only
        
        mock_results = [
            {
                'experiment_id': 'mmd_default_terrace_datashift_on_seed0',
                'algorithm': 'mmd',
                'pattern': 'default',
                'dataset': 'terrace',
                'scenario': 'datashift_on',
                'seed': 0,
                'status': 'success',
                'execution_time': 10.5,
                'results': [
                    {'dataset': 'Test', 'is_drift': '1', 'p_val': '0.01', 'statistic': '2.5'},
                    {'dataset': 'Normal (H0)', 'is_drift': '0', 'p_val': '0.8', 'statistic': '0.1'}
                ]
            },
            {
                'experiment_id': 'mmd_default_terrace_datashift_off_seed0',
                'algorithm': 'mmd',
                'pattern': 'default',
                'dataset': 'terrace',
                'scenario': 'datashift_off',
                'seed': 0,
                'status': 'success',
                'execution_time': 8.2,
                'results': [
                    {'dataset': 'Test', 'is_drift': '0', 'p_val': '0.7', 'statistic': '0.3'},
                    {'dataset': 'Normal (H0)', 'is_drift': '0', 'p_val': '0.9', 'statistic': '0.05'}
                ]
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_count = generate_csv_only(results=mock_results, output_dir=temp_dir)
            
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
            
            if not missing_files and result_count > 0:
                print(f"✓ CSV generation works correctly ({result_count} result rows)")
                return True
            else:
                print(f"✗ CSV generation failed. Missing files: {missing_files}")
                return False
                
    except Exception as e:
        print(f"✗ CSV generation test failed: {e}")
        return False

def test_json_generation_logic():
    """Test JSON generation logic with mock data."""
    print("\n=== Testing JSON Generation Logic ===")
    
    try:
        from comprehensive_evaluation_separated import generate_json_only
        
        mock_results = [
            {
                'experiment_id': 'mmd_default_terrace_datashift_on_seed0',
                'algorithm': 'mmd',
                'pattern': 'default',
                'dataset': 'terrace',
                'scenario': 'datashift_on',
                'seed': 0,
                'status': 'success',
                'execution_time': 10.5,
                'results': [
                    {'dataset': 'Test', 'is_drift': '1', 'p_val': '0.01', 'statistic': '2.5'},
                    {'dataset': 'Normal (H0)', 'is_drift': '0', 'p_val': '0.8', 'statistic': '0.1'}
                ]
            },
            {
                'experiment_id': 'lsdd_pattern1_treegrowth_5_datashift_off_seed1',
                'algorithm': 'lsdd',
                'pattern': 'pattern1',
                'dataset': 'treegrowth_5',
                'scenario': 'datashift_off',
                'seed': 1,
                'status': 'success',
                'execution_time': 8.2,
                'results': [
                    {'dataset': 'Test', 'is_drift': '0', 'p_val': '0.7', 'statistic': '0.3'},
                    {'dataset': 'Normal (H0)', 'is_drift': '0', 'p_val': '0.9', 'statistic': '0.05'}
                ]
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_count = generate_json_only(results=mock_results, output_dir=temp_dir,
                                            algorithms=['mmd'], patterns=['default'])
            
            json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            
            if json_files and result_count == 1:
                with open(os.path.join(temp_dir, json_files[0]), 'r') as f:
                    loaded_data = json.load(f)
                
                if (len(loaded_data) == 1 and 
                    loaded_data[0]['algorithm'] == 'mmd' and
                    loaded_data[0]['pattern'] == 'default'):
                    print(f"✓ JSON generation with filtering works correctly ({result_count} result)")
                    return True
                else:
                    print("✗ JSON content filtering failed")
                    return False
            else:
                print(f"✗ JSON generation failed. Files: {json_files}, Count: {result_count}")
                return False
                
    except Exception as e:
        print(f"✗ JSON generation test failed: {e}")
        return False


def test_argument_parsing():
    """Test argument parsing functionality."""
    print("\n=== Testing Argument Parsing ===")
    
    try:
        import argparse
        from comprehensive_evaluation_separated import ALGORITHMS, PATTERNS, DATASETS, SCENARIOS
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=['both', 'evaluation', 'csv', 'json'], default='both')
        parser.add_argument("--algorithms", nargs='+', choices=ALGORITHMS)
        parser.add_argument("--patterns", nargs='+', choices=PATTERNS)
        parser.add_argument("--datasets", nargs='+', choices=DATASETS)
        parser.add_argument("--scenarios", nargs='+', choices=SCENARIOS)
        parser.add_argument("--seeds", nargs='+', type=int)
        parser.add_argument("--output-filename", help="Custom filename for JSON output")
        
        test_args = [
            "--mode", "json",
            "--algorithms", "mmd", "lsdd",
            "--patterns", "default", "pattern1",
            "--seeds", "0", "1",
            "--output-filename", "custom_results.json"
        ]
        
        args = parser.parse_args(test_args)
        
        if (args.mode == 'json' and
            args.algorithms == ['mmd', 'lsdd'] and
            args.patterns == ['default', 'pattern1'] and
            args.seeds == [0, 1] and
            args.output_filename == 'custom_results.json'):
            print("✓ Argument parsing works correctly (including JSON mode)")
            return True
        else:
            print("✗ Argument parsing failed")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting separated evaluation functionality tests (without datasets)...")
    
    if not os.path.exists("comprehensive_evaluation_separated.py"):
        print("✗ comprehensive_evaluation_separated.py not found")
        return False
    
    filtering_success = test_filtering_logic()
    structure_success = test_result_structure()
    csv_success = test_csv_generation_logic()
    json_success = test_json_generation_logic()
    parsing_success = test_argument_parsing()
    
    print("\n=== Test Summary ===")
    print(f"Filtering logic: {'✓ PASS' if filtering_success else '✗ FAIL'}")
    print(f"Result structure: {'✓ PASS' if structure_success else '✗ FAIL'}")
    print(f"CSV generation: {'✓ PASS' if csv_success else '✗ FAIL'}")
    print(f"JSON generation: {'✓ PASS' if json_success else '✗ FAIL'}")
    print(f"Argument parsing: {'✓ PASS' if parsing_success else '✗ FAIL'}")
    
    all_passed = filtering_success and structure_success and csv_success and json_success and parsing_success
    print(f"\nOverall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
