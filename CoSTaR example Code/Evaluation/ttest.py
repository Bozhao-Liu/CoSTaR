import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

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

# Dataset name = parent folder name (e.g., BRISC)
DATASET_NAME = os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_vals(path):
    """Load a 1D array from txt file with flexible delimiters."""
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


# ---------------------------------------------------------
# Core t-test runner
# ---------------------------------------------------------

def run_ttest_for_metric(metric_name):
    """
    metric_name: 'iou' or 'miss'
    Reads:  Result/prediction/<net>/BCE/<metric_name>.txt
    Writes:
        Result/ttest/p-value/<metric_name>/<DATASET_NAME>.tex
        Result/ttest/t-value/<metric_name>/<DATASET_NAME>.tex
    """
    base_dir = os.path.join("Result", "prediction")
    sub_path = os.path.join("BCE", f"{metric_name}.txt")

    # Collect data per network
    data = {}
    for path in glob.glob(os.path.join(base_dir, "*", sub_path)):
        net = os.path.basename(os.path.dirname(os.path.dirname(path)))  # .../net/BCE/metric.txt
        net_lower = net.lower()
        print(path)
        data[net_lower] = load_vals(path)

    # Order networks by priority list
    networks = [n for n in priority_list if n in data]
    n = len(networks)
    if n == 0:
        print(f"[t-test/{metric_name}] No networks found. Skipping.")
        return

    print(f"[t-test/{metric_name}] Networks: {networks}")

    t_stat = pd.DataFrame(np.zeros((n, n)), index=networks, columns=networks)
    p_val = pd.DataFrame(np.zeros((n, n)), index=networks, columns=networks)

    # Pairwise paired t-tests
    for i in range(n):
        for j in range(n):
            if i == j:
                t_stat.iloc[i, j] = np.nan
                p_val.iloc[i, j] = np.nan
                continue

            a = data[networks[i]]
            b = data[networks[j]]
            m = min(len(a), len(b))
            a = a[:m]
            b = b[:m]

            t, p = ttest_rel(a, b)
            t_stat.iloc[i, j] = t
            p_val.iloc[i, j] = p

    # -------------------------------------------------
    # p-value table: two columns per model (name duplicated)
    # -------------------------------------------------
    colnames = []
    for net in networks:
        colnames.append(net)  # p-value
        colnames.append(net)  # stars

    p_table = pd.DataFrame(index=networks, columns=colnames, dtype=object)

    for i, row_net in enumerate(networks):
        for j, col_net in enumerate(networks):
            p = p_val.iloc[i, j]
            p_str = "--" if np.isnan(p) else f"{p:.3e}"
            s_str = "" if np.isnan(p) else stars(p)

            idx_p = colnames.index(col_net)
            idx_s = colnames.index(col_net, idx_p + 1)

            p_table.iloc[i, idx_p] = p_str
            p_table.iloc[i, idx_s] = s_str

    caption_p = (
        "Pairwise paired $t$-test $p$-values. "
        "(Stars indicate significance: "
        "$^{***}$ for $p<0.001$, "
        "$^{**}$ for $p<0.01$, "
        "$^{*}$ for $p<0.05$.)"
    )

    # Ensure dtype object before LaTeX
    p_table = p_table.astype("object")
    tex_p = p_table.to_latex(
        escape=False,
        column_format="l" + "c" * len(colnames),
        caption=caption_p,
        label=f"tab:ttest_pvalues_{DATASET_NAME}_{metric_name}"
    )

    out_dir_p = os.path.join("Result", "ttest", "p-value", metric_name)
    os.makedirs(out_dir_p, exist_ok=True)
    out_path_p = os.path.join(out_dir_p, f"{DATASET_NAME}.tex")
    with open(out_path_p, "w") as f:
        f.write(tex_p)

    # -------------------------------------------------
    # t-value table
    # -------------------------------------------------
    latex_t = t_stat.copy().astype("object")
    for i in range(n):
        for j in range(n):
            val = t_stat.iloc[i, j]
            latex_t.iloc[i, j] = "--" if np.isnan(val) else f"\\tcell{{{val:.3f}}}"

    caption_t = (
        f"Pairwise paired $t$-test $t$-statistics for {metric_name.upper()} comparisons "
        "between networks (ordered by priority)."
    )

    tex_t = latex_t.to_latex(
        escape=False,
        column_format="l" + "c" * n,
        caption=caption_t,
        label=f"tab:ttest_tvalues_{DATASET_NAME}_{metric_name}"
    )

    out_dir_t = os.path.join("Result", "ttest", "t-value", metric_name)
    os.makedirs(out_dir_t, exist_ok=True)
    out_path_t = os.path.join(out_dir_t, f"{DATASET_NAME}.tex")
    with open(out_path_t, "w") as f:
        f.write(tex_t)

    print(f"[t-test/{metric_name}] Saved .tex to:\n  {out_path_p}\n  {out_path_t}")


# ---------------------------------------------------------
# Run for IoU and miss
# ---------------------------------------------------------


for metric in ["iou", "miss",'BIoU','HD95', 'MSD']:
    run_ttest_for_metric(metric)
