import os
import glob
import numpy as np

# -----------------------------------------
# CONFIG
# -----------------------------------------

MODELS = [
    "unet",
    "deeplabv3",
    "mobilenetv2_unet",
    "mobilenetv3_unet",
    "transunet",
    "unett",
    "segformer",
    "swinunet",
    "medt",
    "missformer",
    "nnunet",
    "medformer",
    "lightawnet",
    "msegnet",
    "jin",
    "jinpp",
    "jinppvit",
    "ujin"
]

METRICS = ["iou", "miss", "BIoU", "HD95", "MSD"]

KDE_POINTS = 50


# -----------------------------------------
# KDE function
# -----------------------------------------

def compute_kde(arr, points=50, low=None, high=None):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]  # drop NaN

    # -------------------------------------------------
    # Detect bounds automatically
    # IoU, miss, BIoU are in [0,1]
    # HD95, MSD are unbounded → no reflection
    # -------------------------------------------------
    if low is None:
        if arr.min() >= 0:
            low = 0.0
        else:
            low = arr.min()

    if high is None:
        high = arr.max()

    # -------------------------------------------------
    # Boundary correction (reflection) ONLY IF bounded
    # -------------------------------------------------
    if low == 0.0 and high == 1.0:
        arr_ref = np.concatenate([
            arr,
            -arr,          # reflect across 0
            2 - arr        # reflect across 1
        ])
    else:
        arr_ref = arr

    n = len(arr_ref)
    if n == 0:
        return np.linspace(low, high, points), np.zeros(points)

    # Scott's bandwidth
    bw = np.std(arr_ref) * n ** (-1 / 5)
    bw = max(bw, 1e-6)

    xs = np.linspace(low, high, points)

    # -------------------------------------------------
    # Vectorized KDE (fast)
    # -------------------------------------------------
    diffs = (xs[:, None] - arr_ref[None, :]) / bw
    ys = np.exp(-0.5 * diffs**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))

    return xs, ys


# -----------------------------------------
# Main loop
# -----------------------------------------

for metric in METRICS:
    out_dir = f"Result/kde/{metric}"
    os.makedirs(out_dir, exist_ok=True)

    for model in MODELS:
        txt_path = f"Result/prediction/{model}/BCE/{metric}.txt"

        if not os.path.exists(txt_path):
            print(f"[skip] Missing: {txt_path}")
            continue

        print(f"[load] {txt_path}")
        # read values
        with open(txt_path) as f:
            text = f.read()
        vals = []
        for tok in text.replace("\n", ",").split(","):
            try:
                if float(tok)>0:
                    vals.append(float(tok))
            except:
                pass

        xs, ys = compute_kde(vals, KDE_POINTS)

        # Save .dat
        dat_path = f"{out_dir}/{model}.dat"
        with open(dat_path, "w") as f:
            for x, y in zip(xs, ys):
                f.write(f"{x:.6f} {y:.6f}\n")

        print(f"[saved] {dat_path}")
