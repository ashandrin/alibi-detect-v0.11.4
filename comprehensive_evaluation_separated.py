#!/usr/bin/env python3
"""
Comprehensive drift detection evaluation script with separated evaluation and CSV generation.
Updated for new dataset structure: 6 algorithms × 5 patterns × 10 seeds × 7 datasets × 2 scenarios = 4200 experiments
(1 terrace + 6 treegrowth variants: treegrowth_5, treegrowth_6, treegrowth_7, treegrowth_8, treegrowth_9, treegrowth_10)

This version separates performance evaluation and CSV generation functionality:
- Run evaluation only: --mode evaluation
- Generate CSV from existing results: --mode csv --results-file <path>
- Run both (default): --mode both

Both functions support filtering by: ALGORITHMS, PATTERNS, DATASETS, SCENARIOS, SEED
"""

import os
import sys
import subprocess
import csv
import time
import json
from pathlib import Path
from itertools import product

HYPERPARAMETER_CONFIGS = {
    'mmd': {
        'default': {'mmd_n_permutations': 100, 'mmd_configure_kernel_from_x_ref': True, 'encoding_dim': 2048},
        'pattern1': {'mmd_n_permutations': 200, 'mmd_sigma': [0.1, 1.0, 10.0], 'mmd_configure_kernel_from_x_ref': False, 'encoding_dim': 2048},
        'pattern2': {'mmd_n_permutations': 500, 'mmd_sigma': None, 'mmd_configure_kernel_from_x_ref': True, 'encoding_dim': 1024},
        'pattern3': {'mmd_n_permutations': 50, 'mmd_sigma': [0.5, 2.0], 'mmd_configure_kernel_from_x_ref': False, 'encoding_dim': 512},
        'pattern4': {'mmd_n_permutations': 300, 'mmd_sigma': [1.0, 5.0, 20.0], 'mmd_configure_kernel_from_x_ref': False, 'encoding_dim': 256}
    },
    'lsdd': {
        'default': {'lsdd_n_permutations': 100, 'lsdd_lambda_rd_max': 0.2, 'encoding_dim': 2048},
        'pattern1': {'lsdd_n_permutations': 200, 'lsdd_n_kernel_centers': 100, 'lsdd_lambda_rd_max': 0.1, 'encoding_dim': 2048},
        'pattern2': {'lsdd_n_permutations': 200, 'lsdd_sigma': 2.0, 'lsdd_lambda_rd_max': 0.2, 'encoding_dim': 1024},
        'pattern3': {'lsdd_n_permutations': 300, 'lsdd_n_kernel_centers': 50, 'lsdd_lambda_rd_max': 0.05, 'encoding_dim': 512},
        'pattern4': {'lsdd_n_permutations': 150, 'lsdd_n_kernel_centers': 30, 'lsdd_lambda_rd_max': 0.2, 'encoding_dim': 256}
    },
    'lmmd': {
        'default': {'lmmd_epochs': 3, 'lmmd_learning_rate': 1e-3, 'lmmd_batch_size': 32, 'encoding_dim': 1024},
        'pattern1': {'lmmd_epochs': 15, 'lmmd_learning_rate': 5e-4, 'lmmd_batch_size': 32, 'lmmd_train_size': 0.8, 'encoding_dim': 1024},
        'pattern2': {'lmmd_epochs': 10, 'lmmd_learning_rate': 1e-3, 'lmmd_batch_size': 32, 'lmmd_var_reg': 1e-4, 'encoding_dim': 512},
        'pattern3': {'lmmd_epochs': 20, 'lmmd_learning_rate': 2e-4, 'lmmd_batch_size': 32, 'lmmd_n_permutations': 200, 'encoding_dim': 256},
        'pattern4': {'lmmd_epochs': 5, 'lmmd_learning_rate': 2e-3, 'lmmd_batch_size': 32, 'lmmd_train_size': 0.9, 'encoding_dim': 512}
    },
    'ks': {
        'default': {'ks_correction': 'bonferroni', 'ks_alternative': 'two-sided', 'encoding_dim': 2048},
        'pattern1': {'ks_correction': 'bonferroni', 'ks_alternative': 'two-sided', 'encoding_dim': 2048},
        'pattern2': {'ks_correction': 'fdr', 'ks_alternative': 'two-sided', 'encoding_dim': 1024},
        'pattern3': {'ks_correction': 'bonferroni', 'ks_alternative': 'greater', 'encoding_dim': 512},
        'pattern4': {'ks_correction': 'fdr', 'ks_alternative': 'greater', 'encoding_dim': 256}
    },
    'cvm': {
        'default': {'cvm_correction': 'bonferroni', 'encoding_dim': 2048},
        'pattern1': {'cvm_correction': 'bonferroni', 'encoding_dim': 2048},
        'pattern2': {'cvm_correction': 'fdr', 'encoding_dim': 1024},
        'pattern3': {'cvm_correction': 'bonferroni', 'encoding_dim': 512},
        'pattern4': {'cvm_correction': 'fdr', 'encoding_dim': 256}
    },
    'spot': {
        'default': {'spot_epochs': 3, 'spot_n_diffs': 1, 'spot_l1_reg': 0.01, 'encoding_dim': 1024},
        'pattern1': {'spot_epochs': 15, 'spot_n_diffs': 3, 'spot_l1_reg': 0.005, 'spot_train_size': 0.8, 'encoding_dim': 1024},
        'pattern2': {'spot_epochs': 10, 'spot_n_diffs': 5, 'spot_l1_reg': 0.01, 'spot_n_folds': 5, 'encoding_dim': 512},
        'pattern3': {'spot_epochs': 20, 'spot_n_diffs': 2, 'spot_l1_reg': 0.001, 'spot_learning_rate': 5e-4, 'encoding_dim': 256},
        'pattern4': {'spot_epochs': 8, 'spot_n_diffs': 4, 'spot_l1_reg': 0.02, 'spot_learning_rate': 2e-3, 'encoding_dim': 512}
    }
}

ALGORITHMS = ['mmd', 'lsdd', 'lmmd', 'ks', 'cvm', 'spot']
PATTERNS = ['default', 'pattern1', 'pattern2', 'pattern3', 'pattern4']
SEEDS = list(range(10))
DATASETS = ['terrace', 'treegrowth_5', 'treegrowth_6', 'treegrowth_7', 'treegrowth_8', 'treegrowth_9', 'treegrowth_10']
SCENARIOS = ['datashift_on', 'datashift_off']


def apply_filters(algorithms=None, patterns=None, datasets=None, scenarios=None, seeds=None):
    """Apply filtering to the parameter lists."""
    filtered_algorithms = algorithms if algorithms else ALGORITHMS
    filtered_patterns = patterns if patterns else PATTERNS
    filtered_datasets = datasets if datasets else DATASETS
    filtered_scenarios = scenarios if scenarios else SCENARIOS
    filtered_seeds = seeds if seeds else SEEDS
    return filtered_algorithms, filtered_patterns, filtered_datasets, filtered_scenarios, filtered_seeds


def build_command_args(algorithm, pattern, dataset, scenario, seed, output_path):
    """Build command line arguments for a specific experiment configuration."""
    config = HYPERPARAMETER_CONFIGS[algorithm][pattern]
    
    base_args = [
        sys.executable,
        "cd_unified_sf_10917_t8.3_768x768_clip.py",
        "--train", f"./dataset/{dataset}/{scenario}/train",
        "--test", f"./dataset/{dataset}/{scenario}/test",
        "--output", output_path,
        "--algorithm", algorithm,
        "--seed", str(seed),
        "--p_val_th", "0.05",
        "--encoding_dim", str(config.get('encoding_dim', 1))
    ]
    
    for param, value in config.items():
        if param == 'encoding_dim':
            continue
        elif value is None:
            continue
        elif isinstance(value, bool):
            if value:
                base_args.append(f"--{param}")
        elif isinstance(value, list):
            base_args.extend([f"--{param}"] + [str(v) for v in value])
        else:
            base_args.extend([f"--{param}", str(value)])
    
    return base_args


def run_single_experiment(algorithm, pattern, dataset, scenario, seed, output_base_dir):
    """Run a single experiment and return the results."""
    experiment_id = f"{algorithm}_{pattern}_{dataset}_{scenario}_seed{seed}"
    algorithm_pattern_dir = os.path.join(output_base_dir, f"{algorithm}_{pattern}")
    output_path = os.path.join(algorithm_pattern_dir, experiment_id)
    
    os.makedirs(output_path, exist_ok=True)
    
    cmd = build_command_args(algorithm, pattern, dataset, scenario, seed, output_path)
    
    print(f"Running experiment: {experiment_id}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            results_csv = os.path.join(output_path, "results", "drift_detection_results.csv")
            if os.path.exists(results_csv):
                with open(results_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    csv_results = list(reader)
                
                return {
                    'experiment_id': experiment_id,
                    'algorithm': algorithm,
                    'pattern': pattern,
                    'dataset': dataset,
                    'scenario': scenario,
                    'seed': seed,
                    'status': 'success',
                    'execution_time': execution_time,
                    'results': csv_results
                }
            else:
                return {
                    'experiment_id': experiment_id,
                    'algorithm': algorithm,
                    'pattern': pattern,
                    'dataset': dataset,
                    'scenario': scenario,
                    'seed': seed,
                    'status': 'failed',
                    'error': f'Results CSV not found: {results_csv}',
                    'execution_time': execution_time
                }
        else:
            return {
                'experiment_id': experiment_id,
                'algorithm': algorithm,
                'pattern': pattern,
                'dataset': dataset,
                'scenario': scenario,
                'seed': seed,
                'status': 'failed',
                'error': f'Script execution failed with return code {result.returncode}',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time
            }
    
    except subprocess.TimeoutExpired:
        return {
            'experiment_id': experiment_id,
            'algorithm': algorithm,
            'pattern': pattern,
            'dataset': dataset,
            'scenario': scenario,
            'seed': seed,
            'status': 'timeout',
            'error': 'Script execution timed out after 600 seconds'
        }
    except Exception as e:
        return {
            'experiment_id': experiment_id,
            'algorithm': algorithm,
            'pattern': pattern,
            'dataset': dataset,
            'scenario': scenario,
            'seed': seed,
            'status': 'error',
            'error': f'Unexpected error: {str(e)}'
        }


def check_dataset_availability():
    """Check which datasets are available."""
    available_datasets = []
    for dataset in DATASETS:
        dataset_available = True
        for scenario in SCENARIOS:
            train_path = f"./dataset/{dataset}/{scenario}/train"
            test_path = f"./dataset/{dataset}/{scenario}/test"
            if not (os.path.exists(train_path) and os.path.exists(test_path)):
                dataset_available = False
                print(f"Dataset {dataset}/{scenario} not available: train={os.path.exists(train_path)}, test={os.path.exists(test_path)}")
                break
        if dataset_available:
            available_datasets.append(dataset)
            print(f"✓ Dataset {dataset} is available")
    
    return available_datasets


def run_evaluation_only(output_dir="./output/comprehensive_results", 
                       algorithms=None, patterns=None, datasets=None, scenarios=None, seeds=None,
                       test_mode=False):
    """Run only the performance evaluation with specified filtering conditions."""
    os.makedirs(output_dir, exist_ok=True)
    
    available_datasets = check_dataset_availability()
    if not available_datasets:
        print("Error: No datasets are available")
        return []
    
    print(f"Available datasets: {available_datasets}")
    
    filtered_algorithms, filtered_patterns, filtered_datasets, filtered_scenarios, filtered_seeds = apply_filters(
        algorithms, patterns, datasets, scenarios, seeds)
    
    filtered_datasets = [d for d in filtered_datasets if d in available_datasets]
    
    if test_mode:
        filtered_datasets = filtered_datasets[:2]
        filtered_algorithms = filtered_algorithms[:2]
        filtered_patterns = filtered_patterns[:2]
        filtered_seeds = filtered_seeds[:1]
        filtered_scenarios = filtered_scenarios[:1]

    total_experiments = len(filtered_algorithms) * len(filtered_patterns) * len(filtered_seeds) * len(filtered_datasets) * len(filtered_scenarios)
    print(f"Starting evaluation with {total_experiments} total experiments")
    print(f"Configuration: {len(filtered_algorithms)} algorithms × {len(filtered_patterns)} patterns × {len(filtered_seeds)} seeds × {len(filtered_datasets)} datasets × {len(filtered_scenarios)} scenarios")
    
    all_results = []
    experiment_count = 0
    
    for algorithm, pattern, dataset, scenario, seed in product(filtered_algorithms, filtered_patterns, filtered_datasets, filtered_scenarios, filtered_seeds):
        experiment_count += 1
        print(f"\n=== Experiment {experiment_count}/{total_experiments} ===")
        
        result = run_single_experiment(algorithm, pattern, dataset, scenario, seed, output_dir)
        all_results.append(result)
        
        if result['status'] == 'success':
            print(f"✓ Success in {result['execution_time']:.2f}s")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    return all_results


def save_evaluation_results(results, output_dir):
    """Save evaluation results to JSON for later CSV generation."""
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {results_file}")
    return results_file


def load_evaluation_results(results_file):
    """Load evaluation results from JSON file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} evaluation results from: {results_file}")
    return results


def filter_results(results, algorithms=None, patterns=None, datasets=None, scenarios=None, seeds=None):
    """Filter results based on specified criteria."""
    filtered_algorithms, filtered_patterns, filtered_datasets, filtered_scenarios, filtered_seeds = apply_filters(
        algorithms, patterns, datasets, scenarios, seeds)
    
    filtered_results = []
    for result in results:
        if (result['algorithm'] in filtered_algorithms and
            result['pattern'] in filtered_patterns and
            result['dataset'] in filtered_datasets and
            result['scenario'] in filtered_scenarios and
            result['seed'] in filtered_seeds):
            filtered_results.append(result)
    
    print(f"Filtered {len(filtered_results)} results from {len(results)} total results")
    return filtered_results


def generate_json_only(results=None, results_file=None, output_dir="./output/comprehensive_results",
                      algorithms=None, patterns=None, datasets=None, scenarios=None, seeds=None,
                      output_filename=None):
    """Generate filtered JSON files from existing evaluation results with specified filtering conditions."""
    if results is None:
        if results_file is None:
            results_file = os.path.join(output_dir, "evaluation_results.json")
        results = load_evaluation_results(results_file)
    
    if any([algorithms, patterns, datasets, scenarios, seeds]):
        results = filter_results(results, algorithms, patterns, datasets, scenarios, seeds)
    
    if not results:
        print("No results match the specified filtering criteria")
        return 0
    
    if output_filename is None:
        filter_parts = []
        if algorithms:
            filter_parts.append(f"alg-{'_'.join(algorithms)}")
        if patterns:
            filter_parts.append(f"pat-{'_'.join(patterns)}")
        if datasets:
            filter_parts.append(f"data-{'_'.join(datasets)}")
        if scenarios:
            filter_parts.append(f"scen-{'_'.join(scenarios)}")
        if seeds:
            filter_parts.append(f"seed-{'_'.join(map(str, seeds))}")
        
        if filter_parts:
            output_filename = f"filtered_results_{'_'.join(filter_parts)}.json"
        else:
            output_filename = "filtered_results_all.json"
    
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Filtered JSON results saved to: {output_path}")
    print(f"Total filtered results: {len(results)}")
    
    if results:
        algorithms_found = set(r['algorithm'] for r in results)
        patterns_found = set(r['pattern'] for r in results)
        datasets_found = set(r['dataset'] for r in results)
        scenarios_found = set(r['scenario'] for r in results)
        seeds_found = set(r['seed'] for r in results)
        
        print(f"Filtered results summary:")
        print(f"  Algorithms: {sorted(algorithms_found)}")
        print(f"  Patterns: {sorted(patterns_found)}")
        print(f"  Datasets: {sorted(datasets_found)}")
        print(f"  Scenarios: {sorted(scenarios_found)}")
        print(f"  Seeds: {sorted(seeds_found)}")
    
    return len(results)


def generate_csv_only(results=None, results_file=None, output_dir="./output/comprehensive_results",
                     algorithms=None, patterns=None, datasets=None, scenarios=None, seeds=None):
    """Generate summary CSV files from existing evaluation results with specified filtering conditions."""
    if results is None:
        if results_file is None:
            results_file = os.path.join(output_dir, "evaluation_results.json")
        results = load_evaluation_results(results_file)
    
    if any([algorithms, patterns, datasets, scenarios, seeds]):
        results = filter_results(results, algorithms, patterns, datasets, scenarios, seeds)
    
    comprehensive_data = []
    
    for result in results:
        if result['status'] == 'success' and 'results' in result:
            for test_result in result['results']:
                comprehensive_data.append({
                    'algorithm': result['algorithm'],
                    'pattern': result['pattern'],
                    'dataset': result['dataset'],
                    'scenario': result['scenario'],
                    'seed': result['seed'],
                    'test_dataset': test_result['dataset'],
                    'is_drift': int(test_result['is_drift']),
                    'p_val': float(test_result['p_val']),
                    'statistic': float(test_result['statistic'])
                })
    
    if comprehensive_data:
        comprehensive_csv = os.path.join(output_dir, "comprehensive_results.csv")
        with open(comprehensive_csv, 'w', newline='') as f:
            fieldnames = ['algorithm', 'pattern', 'dataset', 'scenario', 'seed', 
                         'test_dataset', 'is_drift', 'p_val', 'statistic']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comprehensive_data)
        
        print(f"\nComprehensive results CSV saved to: {comprehensive_csv}")
        
        algorithm_pattern_combinations = set((row['algorithm'], row['pattern']) for row in comprehensive_data)
        for algorithm, pattern in algorithm_pattern_combinations:
            algo_pattern_data = [row for row in comprehensive_data if row['algorithm'] == algorithm and row['pattern'] == pattern]
            algo_pattern_dir = os.path.join(output_dir, f"{algorithm}_{pattern}")
            algo_pattern_csv = os.path.join(algo_pattern_dir, f"{algorithm}_{pattern}_results.csv")
            os.makedirs(algo_pattern_dir, exist_ok=True)
            
            with open(algo_pattern_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(algo_pattern_data)
            
            print(f"Algorithm×Pattern results saved to: {algo_pattern_csv}")
        
        summary_stats = {}
        confusion_matrices = {}
        
        for algorithm, pattern in algorithm_pattern_combinations:
            algo_pattern_key = f"{algorithm}_{pattern}"
            algo_pattern_data = [row for row in comprehensive_data if row['algorithm'] == algorithm and row['pattern'] == pattern]
            
            datashift_on_data = [row for row in algo_pattern_data if row['scenario'] == 'datashift_on' and row['test_dataset'] == 'Test']
            datashift_off_data = [row for row in algo_pattern_data if row['scenario'] == 'datashift_off' and row['test_dataset'] == 'Test']
            
            drift_detected_on = sum(1 for row in datashift_on_data if row['is_drift'] == 1)
            drift_detected_off = sum(1 for row in datashift_off_data if row['is_drift'] == 1)
            
            true_positives = drift_detected_on
            false_positives = drift_detected_off
            true_negatives = len(datashift_off_data) - drift_detected_off
            false_negatives = len(datashift_on_data) - drift_detected_on
            
            total_on = len(datashift_on_data)
            total_off = len(datashift_off_data)
            drift_detection_rate = (drift_detected_on / total_on * 100) if total_on > 0 else 0.0
            false_positive_rate = (drift_detected_off / total_off * 100) if total_off > 0 else 0.0
            
            precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0.0
            recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            accuracy = ((true_positives + true_negatives) / (total_on + total_off)) * 100 if (total_on + total_off) > 0 else 0.0
            
            confusion_matrices[algo_pattern_key] = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
            
            summary_stats[algo_pattern_key] = {
                'total_tests': len(algo_pattern_data),
                'datashift_on_total': total_on,
                'datashift_off_total': total_off,
                'drift_detected_on': drift_detected_on,
                'drift_detected_off': drift_detected_off,
                'drift_detection_rate': round(drift_detection_rate, 2),
                'false_positive_rate': round(false_positive_rate, 2),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1_score': round(f1_score, 2),
                'accuracy': round(accuracy, 2)
            }
            
            print(f"  {algo_pattern_key.upper()}: {len(algo_pattern_data)} tests total")
            print(f"    - datashift_on (Test only): {drift_detected_on}/{total_on} detected ({drift_detection_rate:.2f}%)")
            print(f"    - datashift_off (Test only): {drift_detected_off}/{total_off} false positives ({false_positive_rate:.2f}%)")
            print(f"    - Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
            print(f"    - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1_score:.2f}%, Accuracy: {accuracy:.2f}%")
            
            print(f"    - Confusion Matrix Table for {algo_pattern_key.upper()}:")
            print("              Predicted")
            print("      Actual  |  Drift  | No Drift")
            print("      --------|---------|----------")
            print(f"      Drift   |   {true_positives:2d}    |    {false_negatives:2d}")
            print(f"      No Drift|   {false_positives:2d}    |    {true_negatives:2d}")
            print()
        
        summary_csv = os.path.join(output_dir, "summary_statistics.csv")
        with open(summary_csv, 'w', newline='') as f:
            fieldnames = ['algorithm_pattern', 'total_tests', 'datashift_on_total', 'datashift_off_total', 
                         'drift_detected_on', 'drift_detected_off', 'drift_detection_rate', 'false_positive_rate',
                         'true_positives', 'false_positives', 'true_negatives', 'false_negatives',
                         'precision', 'recall', 'f1_score', 'accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for algo_pattern_key, stats in summary_stats.items():
                writer.writerow({'algorithm_pattern': algo_pattern_key, **stats})
        
        confusion_csv = os.path.join(output_dir, "confusion_matrices.csv")
        with open(confusion_csv, 'w', newline='') as f:
            fieldnames = ['algorithm_pattern', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives',
                         'precision', 'recall', 'f1_score', 'accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for algo_pattern_key in algorithm_pattern_combinations:
                algo_pattern_key_str = f"{algo_pattern_key[0]}_{algo_pattern_key[1]}"
                stats = summary_stats[algo_pattern_key_str]
                writer.writerow({
                    'algorithm_pattern': algo_pattern_key_str,
                    'true_positives': stats['true_positives'],
                    'false_positives': stats['false_positives'],
                    'true_negatives': stats['true_negatives'],
                    'false_negatives': stats['false_negatives'],
                    'precision': stats['precision'],
                    'recall': stats['recall'],
                    'f1_score': stats['f1_score'],
                    'accuracy': stats['accuracy']
                })
        
        confusion_table_csv = os.path.join(output_dir, "confusion_matrix_tables.csv")
        with open(confusion_table_csv, 'w', newline='') as f:
            fieldnames = ['algorithm_pattern', 'actual_drift', 'actual_no_drift']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for algo_pattern_key in algorithm_pattern_combinations:
                algo_pattern_key_str = f"{algo_pattern_key[0]}_{algo_pattern_key[1]}"
                stats = summary_stats[algo_pattern_key_str]
                writer.writerow({
                    'algorithm_pattern': f"{algo_pattern_key_str.upper()}_CONFUSION_MATRIX",
                    'actual_drift': "Predicted_Drift",
                    'actual_no_drift': "Predicted_No_Drift"
                })
                writer.writerow({
                    'algorithm_pattern': "Actual_Drift",
                    'actual_drift': stats['true_positives'],
                    'actual_no_drift': stats['false_negatives']
                })
                writer.writerow({
                    'algorithm_pattern': "Actual_No_Drift",
                    'actual_drift': stats['false_positives'],
                    'actual_no_drift': stats['true_negatives']
                })
                writer.writerow({
                    'algorithm_pattern': "---",
                    'actual_drift': "---",
                    'actual_no_drift': "---"
                })
        
        print(f"Summary statistics CSV saved to: {summary_csv}")
        print(f"Confusion matrices CSV saved to: {confusion_csv}")
        print(f"Confusion matrix tables CSV saved to: {confusion_table_csv}")
        
        return len(comprehensive_data)
    else:
        print("No successful results to save")
        return 0


def save_detailed_results(results, output_dir):
    """Save detailed results including failed experiments."""
    detailed_json = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results JSON saved to: {detailed_json}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive drift detection evaluation with separated functionality")
    parser.add_argument("--output", default="./output/comprehensive_results",
                       help="Output directory for results")
    parser.add_argument("--test", action='store_true', 
                       help="Run test mode with reduced parameter space")
    parser.add_argument("--mode", choices=['both', 'evaluation', 'csv', 'json'], default='both',
                       help="Mode: 'both' (default), 'evaluation' only, 'csv' generation only, or 'json' filtered output only")
    parser.add_argument("--results-file", help="JSON file with evaluation results (for csv mode)")
    parser.add_argument("--algorithms", nargs='+', choices=ALGORITHMS, help="Filter by algorithms")
    parser.add_argument("--patterns", nargs='+', choices=PATTERNS, help="Filter by patterns")
    parser.add_argument("--datasets", nargs='+', choices=DATASETS, help="Filter by datasets")
    parser.add_argument("--scenarios", nargs='+', choices=SCENARIOS, help="Filter by scenarios")
    parser.add_argument("--seeds", nargs='+', type=int, help="Filter by seeds")
    parser.add_argument("--output-filename", help="Custom filename for JSON output (json mode only)")
    
    args = parser.parse_args()
    
    print("=== Comprehensive Drift Detection Evaluation (Separated) ===")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Output directory: {args.output}")
    print(f"Mode: {args.mode}")
    print(f"Test mode: {args.test}")
    
    if args.algorithms:
        print(f"Filtered algorithms: {args.algorithms}")
    if args.patterns:
        print(f"Filtered patterns: {args.patterns}")
    if args.datasets:
        print(f"Filtered datasets: {args.datasets}")
    if args.scenarios:
        print(f"Filtered scenarios: {args.scenarios}")
    if args.seeds:
        print(f"Filtered seeds: {args.seeds}")
    
    results = None
    
    if args.mode in ['both', 'evaluation']:
        print("\n=== Running Evaluation ===")
        results = run_evaluation_only(args.output, args.algorithms, args.patterns, 
                                    args.datasets, args.scenarios, args.seeds, args.test)
        if results:
            save_evaluation_results(results, args.output)
            save_detailed_results(results, args.output)
        else:
            print("No evaluation results generated")
            if args.mode == 'evaluation':
                sys.exit(1)
    
    if args.mode in ['both', 'csv']:
        print("\n=== Generating CSV Files ===")
        try:
            successful_results = generate_csv_only(results, args.results_file, args.output, 
                                                 args.algorithms, args.patterns, args.datasets, 
                                                 args.scenarios, args.seeds)
            
            if results:
                success_rate = sum(1 for r in results if r['status'] == 'success') / len(results) * 100
                print(f"\nOverall success rate: {success_rate:.1f}%")
                print(f"Total successful result rows: {successful_results}")
            else:
                print(f"Total result rows generated: {successful_results}")
                
        except Exception as e:
            print(f"Error generating CSV files: {e}")
            sys.exit(1)
    
    if args.mode == 'json':
        print("\n=== Generating Filtered JSON File ===")
        try:
            filtered_count = generate_json_only(results, args.results_file, args.output,
                                              args.algorithms, args.patterns, args.datasets,
                                              args.scenarios, args.seeds, args.output_filename)
            
            print(f"Total filtered results: {filtered_count}")
            
        except Exception as e:
            print(f"Error generating JSON file: {e}")
            sys.exit(1)
    
    print("\n=== Execution Complete ===")
