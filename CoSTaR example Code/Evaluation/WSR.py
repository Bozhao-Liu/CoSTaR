import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

priority_list = [
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

DATASET_NAME = os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_vals(path):
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        try:
            arr = np.loadtxt(path)
        except ValueError:
            with open(path) as f:
                arr = [list(map(float, line.replace(",", " ").split())) for line in f]
            arr = np.array(arr)
    return arr.flatten()


def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def scale_matrix(mat, digits=1):
    """Scale large W-matrix by 10^X for readability."""
    absmax = np.nanmax(np.abs(mat.values))
    if absmax == 0 or np.isnan(absmax):
        return mat.copy(), 0, ""
    exponent = int(np.floor(np.log10(absmax)))
    if exponent < 4:
        return mat.copy(), 0, ""
    scale = 10 ** exponent
    scaled = (mat / scale).round(digits)
    caption = f"(Values scaled by $10^{{{exponent}}}$.)"
    return scaled, exponent, caption


# ---------------------------------------------------------
# Core Wilcoxon runner
# ---------------------------------------------------------

def run_wsr_for_metric(metric_name):
    """
    metric_name: 'iou' or 'miss'
    Reads:  Result/prediction/<net>/BCE/<metric_name>.txt
    Writes:
        Result/wsr/p-value/<metric_name>/<DATASET_NAME>.tex
        Result/wsr/w-value/<metric_name>/<DATASET_NAME>.tex
    """
    base_dir = os.path.join("Result", "prediction")
    sub_path = os.path.join("BCE", f"{metric_name}.txt")

    data = {}
    for path in glob.glob(os.path.join(base_dir, "*", sub_path)):
        net = os.path.basename(os.path.dirname(os.path.dirname(path)))  # .../net/BCE/metric.txt
        net_lower = net.lower()
        data[net_lower] = load_vals(path)

    networks = [n for n in priority_list if n in data]
    n = len(networks)
    if n == 0:
        print(f"[wsr/{metric_name}] No networks found. Skipping.")
        return

    print(f"[wsr/{metric_name}] Networks: {networks}")

    W_mat = pd.DataFrame(np.zeros((n, n)), index=networks, columns=networks)
    p_mat = pd.DataFrame(np.zeros((n, n)), index=networks, columns=networks)

    m_final = None  # number of paired images

    # Pairwise Wilcoxon
    for i in range(n):
        for j in range(n):
            if i == j:
                W_mat.iloc[i, j] = np.nan
                p_mat.iloc[i, j] = np.nan
                continue

            a = data[networks[i]]
            b = data[networks[j]]
            m = min(len(a), len(b))
            a = a[:m]
            b = b[:m]
            if m_final is None:
                m_final = m

            try:
                W, p = wilcoxon(a, b, zero_method="wilcox", correction=False)
            except ValueError:
                W, p = 0, 1.0

            W_mat.iloc[i, j] = W
            p_mat.iloc[i, j] = p

    # -------------------------------------------------
    # W-value table (scaled)
    # -------------------------------------------------
    W_scaled, exponent, caption_scale = scale_matrix(W_mat, digits=1)

    latex_W = W_scaled.copy().astype("object")
    for i in range(n):
        for j in range(n):
            val = W_scaled.iloc[i, j]
            latex_W.iloc[i, j] = "--" if np.isnan(val) else f"{val:.1f}"

    caption_W = (
        "Pairwise Wilcoxon Signed-Rank $W$-statistics "
        f"{caption_scale} "
        f"(computed over {m_final} paired test images)."
    )

    tex_W = latex_W.to_latex(
        escape=False,
        column_format="l" + "c" * n,
        caption=caption_W,
        label=f"tab:wsr_wvalues_{DATASET_NAME}_{metric_name}"
    )

    out_dir_w = os.path.join("Result", "WSR", "w-value", metric_name)
    os.makedirs(out_dir_w, exist_ok=True)
    out_path_w = os.path.join(out_dir_w, f"{DATASET_NAME}.tex")
    with open(out_path_w, "w") as f:
        f.write(tex_W)

    # -------------------------------------------------
    # p-value table: two columns per model (duplicated names)
    # -------------------------------------------------
    colnames = []
    for net in networks:
        colnames.append(net)
        colnames.append(net)

    p_table = pd.DataFrame(index=networks, columns=colnames, dtype=object)

    for i, row_net in enumerate(networks):
        for j, col_net in enumerate(networks):
            p = p_mat.iloc[i, j]
            p_str = "--" if np.isnan(p) else f"{p:.3e}"
            s_str = "" if np.isnan(p) else stars(p)

            idx_p = colnames.index(col_net)
            idx_s = colnames.index(col_net, idx_p + 1)

            p_table.iloc[i, idx_p] = p_str
            p_table.iloc[i, idx_s] = s_str

    caption_p = (
        "Pairwise Wilcoxon Signed-Rank $p$-values. "
        "(Stars indicate significance: "
        "$^{***}$ for $p<0.001$, "
        "$^{**}$ for $p<0.01$, "
        "$^{*}$ for $p<0.05$.)"
    )

    tex_p = p_table.to_latex(
        escape=False,
        column_format="l" + "c" * len(colnames),
        caption=caption_p,
        label=f"tab:wsr_pvalues_{DATASET_NAME}_{metric_name}"
    )

    out_dir_p = os.path.join("Result", "WSR", "p-value", metric_name)
    os.makedirs(out_dir_p, exist_ok=True)
    out_path_p = os.path.join(out_dir_p, f"{DATASET_NAME}.tex")
    with open(out_path_p, "w") as f:
        f.write(tex_p)

    print(f"[wsr/{metric_name}] Saved .tex to:\n  {out_path_p}\n  {out_path_w}")


# ---------------------------------------------------------
# Run for IoU and miss
# ---------------------------------------------------------

for metric in ["iou", "miss",'BIoU','HD95', 'MSD']:
    run_wsr_for_metric(metric)
