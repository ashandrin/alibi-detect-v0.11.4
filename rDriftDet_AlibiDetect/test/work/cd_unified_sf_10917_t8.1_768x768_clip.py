from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from tqdm import tqdm

from alibi_detect.cd import MMDDrift, KSDrift, CVMDrift, LSDDDrift, SpotTheDiffDrift, LearnedKernelDrift
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.utils.pytorch import DeepKernel
import torch
import torch.nn as nn
from alibi_detect.cd.pytorch import preprocess_drift

parser = argparse.ArgumentParser(description="Drift detection using various algorithms")
parser.add_argument("--train", type=str, required=True, help="Path to training images directory")
parser.add_argument("--test", type=str, required=True, help="Path to test images directory")
parser.add_argument("--output", type=str, required=True, help="Path to output directory for results")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--patch_coords", type=int, nargs=2, default=[70, 170],
                    help="Patch extraction coordinates (x, y)")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Proportion of training data to use for validation (0.0-1.0)")
parser.add_argument("--encoding_dim", type=int, default=1,
                    help="Dimension of the encoder for preprocessing")
parser.add_argument("--p_val_th", type=float, default=0.05,
                    help="p-value threshold for drift detection")
parser.add_argument("--algorithm", type=str, choices=["mmd", "ks", "cvm", "lsdd", "spot", "lmmd"], default="mmd",
                    help="Drift detection algorithm to use (mmd, ks, cvm, lsdd, spot, or lmmd)")
args = parser.parse_args()

train_path = args.train
test_path = args.test
output_dir = args.output

detector_path = os.path.join(output_dir, "detector_pt")
results_dir = os.path.join(output_dir, "results")
sample_images_dir = os.path.join(output_dir, "sample_images")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(sample_images_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def extract_patch_at_coords(img, coords, patch_size=(768, 768)):
    """
    画像から指定座標のパッチを抽出する

    Args:
        img: 入力画像 (H, W, C)
        coords: パッチの左上座標 (x, y)
        patch_size: 抽出するパッチのサイズ (height, width)

    Returns:
        抽出されたパッチ
    """
    h, w = img.shape[0], img.shape[1]
    patch_h, patch_w = patch_size
    start_w, start_h = coords

    if start_h < 0 or start_w < 0 or start_h + patch_h > h or start_w + patch_w > w:
        raise ValueError(f"Coordinates ({start_w}, {start_h}) with patch size {patch_h}x{patch_w} exceed image dimensions {h}x{w}")

    patch = img[start_h:start_h+patch_h, start_w:start_w+patch_w]

    return patch, (start_w, start_h)

def load_images_from_directory(directory, img_size=(1280, 960), patch_size=(768, 768), patch_coords=(0, 0)):
    """
    指定されたディレクトリから画像を読み込み、パッチを抽出してnumpy配列に変換する

    Args:
        directory: 画像が格納されているディレクトリ
        img_size: 画像のリサイズサイズ
        patch_size: 抽出するパッチのサイズ
        patch_coords: パッチの左上座標 (x, y)

    Returns:
        画像の配列とパッチ位置の配列
    """
    if not os.path.exists(directory):
        print(f"Warning: {directory} does not exist")
        return None, None

    image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                 glob.glob(os.path.join(directory, "*.jpeg")) + \
                 glob.glob(os.path.join(directory, "*.png"))

    if not image_files:
        print(f"Warning: No images found in {directory}")
        return None, None

    images = []
    positions = []  # パッチ位置

    for img_path in tqdm(image_files, desc=f"Loading images from {os.path.basename(directory)}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)  # 1280x960にリサイズ
            img_array = np.array(img) / 255.0  # 正規化

            patch, position = extract_patch_at_coords(img_array, patch_coords, patch_size)

            images.append(patch)
            positions.append(position)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    if images:
        images_array = np.array(images, dtype=np.float32)
        print(f"Loaded {len(images)} images from {directory}, patch size: {patch_size}, position: {patch_coords}")
        return images_array, positions
    else:
        return None, None

print("Loading images...")
patch_size = (768, 768)  # 抽出するパッチのサイズを768x768に変更

patch_coords = tuple(args.patch_coords)  # コマンドライン引数から取得
print(f"Using patch coordinates: {patch_coords}")

X_train, train_positions = load_images_from_directory(
    train_path,
    img_size=(1280, 960),
    patch_size=patch_size,
    patch_coords=patch_coords
)

if X_train is None:
    raise ValueError("No training images found")

X_test, test_positions = load_images_from_directory(
    test_path,
    img_size=(1280, 960),
    patch_size=patch_size,
    patch_coords=patch_coords
)

if X_test is None:
    raise ValueError("No test images found")

test_size = args.test_size
X_ref, X_h0 = train_test_split(X_train, test_size=test_size, random_state=seed)
print(f"X_ref shape: {X_ref.shape}, X_h0 shape: {X_h0.shape}")

X_c = [X_test]
corruption = ["Test"]
print(f"Added test data with shape {X_test.shape}")

def save_sample_images(output_dir=sample_images_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title("Reference (Normal)")
    plt.axis("off")
    plt.imshow(X_ref[0])
    plt.savefig(os.path.join(output_dir, "reference_sample.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title("H0 (Normal)")
    plt.axis("off")
    plt.imshow(X_h0[0])
    plt.savefig(os.path.join(output_dir, "h0_sample.png"), bbox_inches="tight")
    plt.close()

    for i, (x_corr, corr_name) in enumerate(zip(X_c, corruption)):
        plt.figure(figsize=(10, 8))
        plt.title(corr_name)
        plt.axis("off")
        plt.imshow(x_corr[0])
        plt.savefig(os.path.join(output_dir, f"{corr_name}_sample.png"), bbox_inches="tight")
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].set_title("Reference (Normal)")
    axs[0].imshow(X_ref[0])
    axs[0].axis("off")

    axs[1].set_title("H0 (Normal)")
    axs[1].imshow(X_h0[0])
    axs[1].axis("off")

    axs[2].set_title("Test")
    axs[2].imshow(X_c[0][0])
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_samples.png"), bbox_inches="tight")
    plt.close()

    print(f"Sample images saved to {output_dir}")

save_sample_images()

def permute_c(x):
    return np.transpose(x.astype(np.float32), (0, 3, 1, 2))

X_ref_pt = permute_c(X_ref)
X_h0_pt = permute_c(X_h0)
X_c_pt = [permute_c(xc) for xc in X_c]
print(f"PyTorch shapes - X_ref: {X_ref_pt.shape}, X_h0: {X_h0_pt.shape}, X_c[0]: {X_c_pt[0].shape}")

encoding_dim = args.encoding_dim
print(f"Using encoding dimension: {encoding_dim}")

encoder_net = nn.Sequential(
    nn.Conv2d(3, 64, 8, stride=4, padding=0),  # 出力サイズ: (768-8)/4+1 = 191
    nn.ReLU(),
    nn.Conv2d(64, 128, 8, stride=4, padding=0),  # 出力サイズ: (191-8)/4+1 = 46
    nn.ReLU(),
    nn.Conv2d(128, 256, 6, stride=3, padding=0),  # 出力サイズ: (46-6)/3+1 = 14
    nn.ReLU(),
    nn.Conv2d(256, 512, 4, stride=2, padding=0),  # 出力サイズ: (14-4)/2+1 = 6
    nn.ReLU(),
    nn.Flatten(),  # 出力サイズ: 512 * 6 * 6 = 18432
    nn.Linear(512 * 6 * 6, 1024),
    nn.ReLU(),
    nn.Linear(1024, encoding_dim)
).to(device).eval()

preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=32)  # バッチサイズを小さくして大きな画像に対応

p_val_th = args.p_val_th
print(f"Using p-value threshold: {p_val_th}")

algorithm = args.algorithm.lower()
print(f"Using drift detection algorithm: {algorithm}")

def lsdd_preprocess_fn(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return preprocess_fn(x)

statistic_name = ""
if algorithm == "mmd":
    statistic_name = "MMD distance"
    cd = MMDDrift(X_ref_pt, backend="pytorch", p_val=p_val_th, preprocess_fn=preprocess_fn, n_permutations=100)
    filepath = os.path.join(output_dir, "detector_pt")
    os.makedirs(filepath, exist_ok=True)
    save_detector(cd, filepath)
    cd = load_detector(filepath)
elif algorithm == "ks":
    statistic_name = "KS statistic"
    cd = KSDrift(X_ref_pt, p_val=p_val_th, preprocess_fn=preprocess_fn)
    filepath = os.path.join(output_dir, "detector_pt")
    os.makedirs(filepath, exist_ok=True)
    save_detector(cd, filepath)
    cd = load_detector(filepath)
elif algorithm == "cvm":
    statistic_name = "CVM statistic"
    cd = CVMDrift(X_ref_pt, p_val=p_val_th, preprocess_fn=preprocess_fn)
    filepath = os.path.join(output_dir, "detector_pt")
    os.makedirs(filepath, exist_ok=True)
    save_detector(cd, filepath)
    cd = load_detector(filepath)
elif algorithm == "lsdd":
    statistic_name = "LSDD statistic"
    cd = LSDDDrift(
        X_ref_pt, 
        backend="pytorch", 
        p_val=p_val_th, 
        preprocess_fn=lsdd_preprocess_fn,  # カスタム前処理関数を使用
        n_permutations=100
    )
    print("Skipping detector saving for LSDD algorithm to avoid device mismatch errors")
elif algorithm == "spot":
    statistic_name = "Spot-the-diff distance"
    cd = SpotTheDiffDrift(
        X_ref_pt,
        backend="pytorch",
        p_val=p_val_th,
        preprocess_fn=preprocess_fn,
        n_diffs=200,  # 検出する差異の数を増加（50→100→200）
        train_size=0.6,  # トレーニングに使用するデータの割合を減少（0.75→0.6）
        kernel=None,  # デフォルトのカーネルを使用
        l1_reg=0.005,  # L1正則化パラメータを減少（0.01→0.005）
        epochs=5,     # トレーニングエポック数を増加（3→5）
        verbose=1     # 詳細な出力
    )
    # print("Skipping detector saving for Spot-the-diff algorithm to avoid device mismatch errors")
    filepath = os.path.join(output_dir, "detector_pt")
    os.makedirs(filepath, exist_ok=True)
    save_detector(cd, filepath)
    cd = load_detector(filepath)

elif algorithm == "lmmd":
    statistic_name = "LMMD distance"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    proj = nn.Sequential(
        nn.Conv2d(3, 32, 8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, 8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 128, 6, stride=3, padding=0),
        nn.ReLU(),
        nn.Conv2d(128, 256, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256 * 6 * 6, 512),
        nn.ReLU(),
        nn.Linear(512, encoding_dim)
    ).to(device)
    
    kernel = DeepKernel(proj, eps=0.01)
    
    cd = LearnedKernelDrift(
        X_ref_pt, 
        kernel=kernel,
        backend="pytorch", 
        p_val=p_val_th, 
        preprocess_fn=preprocess_fn,
        epochs=10,
        batch_size=32,
        verbose=1,
        device=device
    )
    filepath = os.path.join(output_dir, "detector_pt")
    os.makedirs(filepath, exist_ok=True)
    save_detector(cd, filepath)
    cd = load_detector(filepath)

else:
    raise ValueError(f"Unknown algorithm: {algorithm}")

def make_predictions(cd, x_h0, x_corr, corruption):
    from timeit import default_timer as timer

    results = []
    labels = ["No!", "Yes!"]

    t = timer()
    preds = cd.predict(x_h0)
    dt = timer() - t

    statistic = preds["data"]["distance"]
    p_val = preds["data"]["p_val"]

    if isinstance(p_val, np.ndarray):
        p_val = float(p_val[0])
    if isinstance(statistic, np.ndarray):
        statistic = float(statistic[0])

    print("No degradation")
    print(f"Drift? {labels[preds['data']['is_drift']]}")
    print(f"p-value: {p_val:.3f}")
    print(f"{statistic_name}: {statistic:.3f}")
    print(f"Time (s) {dt:.3f}")

    results.append({
        "dataset": "Normal (H0)",
        "is_drift": preds["data"]["is_drift"],
        "p_val": p_val,
        "statistic": statistic,
        "time": dt
    })

    if isinstance(x_corr, list):
        for x, c in zip(x_corr, corruption):
            t = timer()
            preds = cd.predict(x)
            dt = timer() - t

            statistic = preds["data"]["distance"]
            p_val = preds["data"]["p_val"]

            if isinstance(p_val, np.ndarray):
                p_val = float(p_val[0])
            if isinstance(statistic, np.ndarray):
                statistic = float(statistic[0])

            print("")
            print(f"Test data: {c}")
            print(f"Drift? {labels[preds['data']['is_drift']]}")
            print(f"p-value: {p_val:.3f}")
            print(f"{statistic_name}: {statistic:.3f}")
            print(f"Time (s) {dt:.3f}")

            results.append({
                "dataset": c,
                "is_drift": preds["data"]["is_drift"],
                "p_val": p_val,
                "statistic": statistic,
                "time": dt
            })

    return results

print("\nRunning drift detection...")
results = make_predictions(cd, X_h0_pt, X_c_pt, corruption)

def visualize_and_save_results(results, output_dir=results_dir, p_threshold=args.p_val_th):
    os.makedirs(output_dir, exist_ok=True)

    import csv
    with open(os.path.join(output_dir, "drift_detection_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "is_drift", "p_val", "statistic", "time"])
        writer.writeheader()
        writer.writerows(results)

    datasets = [result["dataset"] for result in results]
    p_vals = [result["p_val"] for result in results]
    statistics = [result["statistic"] for result in results]
    is_drift = [result["is_drift"] for result in results]

    colors = ["green" if not drift else "red" for drift in is_drift]

    plt.figure(figsize=(8, 6))

    x_pos = list(range(len(datasets)))

    bars = plt.bar(x_pos, p_vals, color=colors, width=0.4)
    plt.axhline(y=p_threshold, color="black", linestyle="--", label=f"p={p_threshold} threshold")
    plt.ylabel("p-value")
    plt.title("Drift Detection Results - p-value")

    plt.xticks(x_pos, datasets, rotation=45)

    plt.xlim(-0.5, len(x_pos) - 0.5)

    plt.ylim(0, 1.1)  # y軸の範囲を0-1.1に固定


    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "drift_detection_pvalue.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))

    bars = plt.bar(x_pos, statistics, color=colors, width=0.4)
    plt.ylabel(statistic_name)
    plt.title(f"Drift Detection Results - {statistic_name}")

    plt.xticks(x_pos, datasets, rotation=45)

    plt.xlim(-0.5, len(x_pos) - 0.5)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "drift_detection_statistic.png"), bbox_inches="tight")
    plt.close()

    print(f"Results saved to {output_dir}")

visualize_and_save_results(results)
