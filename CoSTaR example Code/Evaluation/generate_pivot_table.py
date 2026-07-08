import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon



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
    "ujin"
]

metrics = ["iou", "miss", "BIoU", "HD95", "MSD"]

REFERENCE_MODEL = "jinppvit"

base_pred_dir = os.path.join("Result", "prediction")


# ============================================================
# HELPERS
# ============================================================

def load_vals(path):
    """Load a 1D array from .txt with flexible delimiters."""
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
    """Same scaling logic as your original WSR.py."""
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


# ============================================================
# LOAD METRIC DATA
# ============================================================

def load_metric_data(metric):
    data = {}

    # Load reference model (JinPPViT)
    ref_file = os.path.join(base_pred_dir, REFERENCE_MODEL, "BCE", f"{metric}.txt")
    if os.path.exists(ref_file):
        data[REFERENCE_MODEL] = load_vals(ref_file)

    # Load models
    for model in priority_list:
        fpath = os.path.join(base_pred_dir, model, "BCE", f"{metric}.txt")
        if os.path.exists(fpath):
            data[model] = load_vals(fpath)

    return data


# ============================================================
# STATISTICS (t-test & WSR)
# ============================================================

def compute_stats(metric, data):
    if REFERENCE_MODEL not in data:
        raise RuntimeError(f"Reference model '{REFERENCE_MODEL}' missing for metric={metric}")

    ref = data[REFERENCE_MODEL]

    t_val = {}
    t_sig = {}
    w_val = {}
    w_sig = {}

    for model in priority_list:
        if model not in data:
            t_val[model] = None
            t_sig[model] = ""
            w_val[model] = None
            w_sig[model] = ""
            continue

        arr = data[model]
        m = min(len(arr), len(ref))
        a = arr[:m]
        b = ref[:m]

        # ------- t-test -------
        t, p = ttest_rel(a, b)
        t_val[model] = t
        t_sig[model] = f"$p={p:.3f}$" if p > 0.05 else stars(p)

        # ------- Wilcoxon -------
        try:
            W, p2 = wilcoxon(a, b, zero_method="wilcox", correction=False)
        except ValueError:
            W, p2 = np.nan, 1.0
        w_val[model] = W

        if p2 < 0.05:
            w_sig[model] = stars(p2)
        else:
            w_sig[model] = f"$p={p2:.3f}$" if p2 > 0.05 else stars(p2)

    return t_val, t_sig, w_val, w_sig


# ============================================================
# BUILD PIVOT TABLE
# ============================================================

def build_pivot_table():

    # Build column structure
    col_blocks = []
    for metric in metrics:
        col_blocks += [
            f"{metric}_t", f"{metric}_tsig",
            f"{metric}_wsr", f"{metric}_wsrsig"
        ]

    df = pd.DataFrame(index=priority_list, columns=col_blocks)
    wsr_exponents = {}

    for metric in metrics:
        data = load_metric_data(metric)
        t_vals, t_sigs, w_vals, w_sigs = compute_stats(metric, data)

        # WSR scaling
        W_df = pd.DataFrame(
            {model: w_vals[model] for model in priority_list},
            index=["W"]
        ).T

        W_scaled_df, exponent, caption_scale = scale_matrix(W_df, digits=1)
        wsr_exponents[metric] = exponent

        # Fill
        for model in priority_list:

            # t-test wrapped with \tcell{}
            tv = -t_vals[model]
            df.loc[model, f"{metric}_t"] = (
                "--" if tv is None else f"\\tcell{{{tv:.1f}}}"
            )
            df.loc[model, f"{metric}_tsig"] = t_sigs[model]

            # scaled WSR
            wv = W_scaled_df.loc[model, "W"]
            df.loc[model, f"{metric}_wsr"] = (
                "--" if pd.isna(wv) else f"{wv:.1f}"
            )
            df.loc[model, f"{metric}_wsrsig"] = w_sigs[model]

    return df, wsr_exponents


# ============================================================
# LATEX PIVOT TABLE GENERATOR
# ============================================================

def to_latex(df, wsr_exponents):

    # Construct caption text for WSR scaling
    scaling_caption = []
    for metric in metrics:
        exp = wsr_exponents[metric]
        if exp >= 4:
            scaling_caption.append(f"{metric}: $10^{{{exp}}}$")
    if scaling_caption:
        caption_tail = "WSR scaled by " + ", ".join(scaling_caption) + "."
    else:
        caption_tail = "WSR values are unscaled."

    # Header rows
    header1 = "& "
    header2 = "& "
    for metric in metrics:
        header1 += f"\\multicolumn{{4}}{{c|}}{{{metric}}} & "
        header2 += "t & Sig. & WSR & Sig. & "
    header1 = header1.rstrip("& ") + " \\\\"
    header2 = header2.rstrip("& ") + " \\\\"

    # Table lines
    lines = []
    lines.append("% Auto-generated pivot table")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Paired $t$-test and Wilcoxon comparisons vs JinPPViT. " + caption_tail + "}")
    lines.append("\\begin{tabular}{l|" + "c" * (len(metrics) * 4) + "}")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append("\\midrule")
    lines.append(header2)
    lines.append("\\midrule")

    for model in df.index:
        row = model
        for col in df.columns:
            row += " & " + str(df.loc[model, col])
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ============================================================
# EXECUTION
# ============================================================

df, wsr_exponents = build_pivot_table()
latex = to_latex(df, wsr_exponents)

out_dir = os.path.join("Result", "pivot_table")
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, f"{os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))}.tex")
with open(out_file, "w") as f:
    f.write(latex)

print(f"Saved pivot table to:\n  {out_file}")