# Separated Comprehensive Evaluation Script

This document describes the separated comprehensive drift detection evaluation functionality that allows running performance evaluation and CSV generation independently.

## Overview

The `comprehensive_evaluation_separated.py` script separates the previously monolithic evaluation process into two independent functions:

1. **Performance Evaluation Only** - Runs experiments and saves results to JSON
2. **CSV Generation Only** - Generates summary CSV files from existing evaluation results

Both functions support filtering by: ALGORITHMS, PATTERNS, DATASETS, SCENARIOS, SEED

## Usage Examples

### Run Both Evaluation and CSV Generation (Default)
```bash
python comprehensive_evaluation_separated.py --output ./results
```

### Run Only Performance Evaluation
```bash
python comprehensive_evaluation_separated.py --mode evaluation --output ./results
```

### Generate CSV from Existing Results
```bash
python comprehensive_evaluation_separated.py --mode csv --results-file ./results/evaluation_results.json --output ./results
```

### Generate Filtered JSON from Existing Results
```bash
python comprehensive_evaluation_separated.py --mode json --results-file ./results/evaluation_results.json --algorithms mmd lsdd --patterns default
```

### Run with Filtering
```bash
# Evaluate only specific algorithms and patterns
python comprehensive_evaluation_separated.py --mode evaluation --algorithms mmd lsdd --patterns default pattern1

# Generate CSV for specific datasets and scenarios
python comprehensive_evaluation_separated.py --mode csv --datasets terrace treegrowth_5 --scenarios datashift_on
```

## Filtering Parameters

All filtering parameters work with evaluation, CSV generation, and JSON generation modes:

- `--algorithms`: Filter by algorithms (choices: mmd, lsdd, lmmd, ks, cvm, spot)
- `--patterns`: Filter by patterns (choices: default, pattern1, pattern2, pattern3, pattern4)
- `--datasets`: Filter by datasets (choices: terrace, treegrowth_5, treegrowth_6, treegrowth_7, treegrowth_8, treegrowth_9, treegrowth_10)
- `--scenarios`: Filter by scenarios (choices: datashift_on, datashift_off)
- `--seeds`: Filter by seeds (integers 0-9)

## Output Files

### Evaluation Mode
- `evaluation_results.json`: Complete evaluation results for later CSV generation
- `detailed_results.json`: Detailed results including failed experiments

### CSV Mode
- `comprehensive_results.csv`: Overall comprehensive results
- `{algorithm}_{pattern}_results.csv`: Algorithm×pattern-specific results
- `summary_statistics.csv`: Summary statistics and metrics
- `confusion_matrices.csv`: Confusion matrix data
- `confusion_matrix_tables.csv`: Confusion matrices in table format

### JSON Mode
- `filtered_results_{filter_description}.json`: Filtered evaluation results based on specified criteria
- Custom filename supported via `--output-filename` parameter

## Benefits

1. **Resilience**: If evaluation is interrupted, results are saved and CSV/JSON generation can be run separately
2. **Flexibility**: Generate different CSV summaries or filtered JSON files from the same evaluation results using different filters
3. **Efficiency**: Re-generate CSV/JSON files without re-running expensive evaluations
4. **Filtering**: Apply different filtering criteria to evaluation vs CSV/JSON generation
5. **Custom JSON Output**: Create filtered JSON files with specific combinations of parameters for targeted analysis

## Usage Examples for JSON Mode

```bash
# Generate JSON with specific algorithms and patterns
python comprehensive_evaluation_separated.py --mode json --algorithms mmd lsdd --patterns default pattern1

# Generate JSON for specific dataset and scenario combinations
python comprehensive_evaluation_separated.py --mode json --datasets terrace treegrowth_5 --scenarios datashift_on

# Generate JSON with custom filename
python comprehensive_evaluation_separated.py --mode json --algorithms mmd --output-filename mmd_only_results.json

# Generate JSON from existing results with multiple filters
python comprehensive_evaluation_separated.py --mode json --results-file ./results/evaluation_results.json --algorithms mmd lsdd --seeds 0 1 2
```

## Backward Compatibility

The script maintains full backward compatibility with the original implementation:
- All hyperparameter configurations are preserved
- All output formats remain identical
- Default behavior (--mode both) produces the same results as the original script
