#!/bin/bash

# 共通パラメータの設定
SCRIPT="cd_unified_sf_10917_t8.1_768x768_clip.py"
COMMON_ARGS="--seed 42 --patch_coords 70 170 --test_size 0.2 --p_val 0.05"

# アルゴリズム別のencoding_dim設定
declare -A ENCODING_DIMS
ENCODING_DIMS["mmd"]=1
ENCODING_DIMS["ks"]=1
ENCODING_DIMS["cvm"]=1
ENCODING_DIMS["lsdd"]=1
ENCODING_DIMS["spot"]=4
ENCODING_DIMS["lmmd"]=4

# データセットとアルゴリズムの定義
ALGORITHMS=("mmd" "ks" "cvm" "lsdd" "spot" "lmmd")
DATASETS=(
    "treegrowth/datashift_on"
    "treegrowth/datashift_off" 
    "terrace/datashift_on"
    "terrace/datashift_off"
)

# 実行関数
run_experiment() {
    local algorithm=$1
    local dataset=$2
    local test_dir=$3
    local test_name=$4
    
    local train_path="./dataset/${dataset}/train"
    local test_path="./dataset/${dataset}/${test_dir}"
    local output_path="./output/unified/${algorithm}/${dataset}/${test_name}"
    local encoding_dim=${ENCODING_DIMS[$algorithm]}
    
    echo "Running ${algorithm} on ${dataset}/${test_name}..."
    python3 $SCRIPT \
        --train "$train_path" \
        --test "$test_path" \
        --output "$output_path" \
        --algorithm "$algorithm" \
        --encoding_dim "$encoding_dim" \
        $COMMON_ARGS
}

# メイン実行ループ
for algorithm in "${ALGORITHMS[@]}"; do
    echo "=== Running drift detection with ${algorithm^^} algorithm ==="
    
    for dataset in "${DATASETS[@]}"; do
        case "$dataset" in
            "treegrowth"*)
                # treegrowthデータセットの場合、test_5からtest_10まで実行
                for i in {5..10}; do
                    run_experiment "$algorithm" "$dataset" "test_$i" "$i"
                done
                ;;
            "terrace"*)
                # terraceデータセットの場合、testディレクトリのみ
                run_experiment "$algorithm" "$dataset" "test" ""
                ;;
        esac
    done
    echo
done

echo "All experiments completed!"
