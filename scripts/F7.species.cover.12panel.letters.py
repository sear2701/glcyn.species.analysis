import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
path = "/workspaces/glcyn.species.analysis/data/species.cover.site.csv"
df = pd.read_csv(path)

# Variables (12 panels, in order)
vars_12 = [
    "BROTEC", "BRORUB", "POLMON", "PHRAUS",
    "SACRAV", "SALTRA", "BACSALF", "BACSALN",
    "SALEXI", "TAMRAM", "SALGOO", "POPFRE"
]

# Optional: nicer panel titles (edit as you like)
titles = [
    "Bromus tectorum", "Bromus madritensis", "Polypogon monospeliensis", "Phragmites australis",
    "Saccharum ravennae", "Salsola tragus", "Baccharis salicifolia", "Baccharis salicina",
    "Salix exigua", "Tamarix ramosissma", "Salix gooddingii", "Populus fremontii"
]

# -----------------------------
# Summarize mean + 1 SE by ageclassn
# -----------------------------
agg = {v: ["mean", "std", "count"] for v in vars_12}
summ = df.groupby("ageclassn").agg(agg)

# Flatten columns -> var_mean, var_std, var_count
summ.columns = [f"{c0}_{c1}" for (c0, c1) in summ.columns]
summ = summ.reset_index().sort_values("ageclassn")

x = summ["ageclassn"].to_numpy()

def mean_and_se(var: str):
    m = summ[f"{var}_mean"].to_numpy(dtype=float)
    sd = summ[f"{var}_std"].to_numpy(dtype=float)
    n = summ[f"{var}_count"].to_numpy(dtype=float)
    se = sd / np.sqrt(n)
    return m, se

# -----------------------------
# Helper: add label above/below point AND error bar (by ageclassn)
# -----------------------------
def idx_at_x(x_arr, x_value):
    hits = np.where(x_arr == x_value)[0]
    if len(hits) == 0:
        raise ValueError(f"ageclassn={x_value} not found in x: {x_arr}")
    return int(hits[0])

def add_label_at_errorbar(ax, x_arr, y_arr, yerr_arr, x_value, text,
                          where="above", ypad_frac=0.03, fontsize=8, dx_pts=0):
    """
    Places bold label just above/below the error bar for the point at x_value.
    """
    i = idx_at_x(x_arr, x_value)

    y0, y1 = ax.get_ylim()
    ypad = ypad_frac * (y1 - y0)

    if where.lower() == "above":
        y_text = y_arr[i] + yerr_arr[i] + ypad
        va = "bottom"
    elif where.lower() == "below":
        y_text = y_arr[i] - yerr_arr[i] - ypad
        va = "top"
    else:
        raise ValueError("where must be 'above' or 'below'")

    ax.annotate(
        text,
        (x_arr[i], y_text),
        xytext=(dx_pts, 0),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=fontsize,
        fontweight="bold"
    )

# -----------------------------
# Significance label specification
# keys: var -> list of (ageclassn, text, where)
# -----------------------------
sig = {
    "BROTEC": [
        (1,  "a",  "below"),
        (2,  "b", "above"),
        (4,  "b", "above"),
        (6,  "c",  "above"),
        (12, "bc", "above"),
        (25, "c",  "above"),
        (40, "bc", "above"),
        (50, "bc", "above"),
    ],
    "BRORUB": [
        (1,  "a",  "above"),
        (4,  "bc", "above"),
        (6,  "ab", "below"),
        (12, "c*", "above"),
        (25, "c",  "above"),
        (40, "c",  "above"),
        (50, "c",  "above"),
    ],
    "POLMON": [
        (1,  "a",   "above"),
        (2,  "ab",  "below"),
        (4,  "ab", "above"),
        (6,  "bc",  "below"),
        (12, "c",   "above"),
        (25, "c",   "above"),
        (40, "c",   "above"),
        (50, "c",   "above"),
    ],
    "SALTRA": [
        (1,  "a",   "above"),
        (2,  "a",  "above"),
        (4,  "b", "above"),
        (6,  "bc",  "above"),
        (12, "bc",  "above"),
        (25, "c",   "above"),
        (40, "c",   "above"),
        (50, "c",   "above"),
    ],
    "SALEXI": [
        (1,  "a",   "above"),
        (2,  "ab",  "below"),
        (4,  "bc", "above"),
        (6,  "bc", "below"),
        (12, "bc",  "below"),
        (25, "c",   "below"),
    ],
    "TAMRAM": [
        (1,  "a",  "above"),
        (2,  "a",  "below"),
        (4,  "ab", "above"),
        (12, "c",  "above"),
        (25, "bc", "above"),
    ],
    "POPFRE": [
        (1,  "a",   "above"),
        (2,  "a",  "below"),
        (25, "b*c", "above"),
        (40, "c",   "above"),
        (50, "c",   "above"),
    ],
}

# -----------------------------
# Figure setup (4x3 = 12 panels)
# -----------------------------
fig, axes = plt.subplots(
    nrows=4, ncols=3,
    figsize=(9, 11),
    sharex=True,
    sharey=True
)

# Styling similar to example
line_kw = dict(color="black", linewidth=1.5, marker="o", markersize=5)
err_kw = dict(capsize=0)  # no caps

# Plot each panel
for i, var in enumerate(vars_12):
    r, c = divmod(i, 3)
    ax = axes[r, c]

    m, se_arr = mean_and_se(var)

    ax.errorbar(
        x, m, yerr=se_arr,
        **err_kw,
        **line_kw
    )

    ax.set_title(titles[i], fontsize=11, fontweight="bold", pad=4)

    # Add significance labels for selected panels
    if var in sig:
        # Ensure we have stable ylim before placing labels (important w/ shared axes)
        # (sharedy=True means ylim will converge; using current is fine here)
        for (age, txt, where) in sig[var]:
            # If an ageclassn isn't present in the data, skip silently
            if age not in x:
                continue
            add_label_at_errorbar(
                ax,
                x_arr=x,
                y_arr=m,
                yerr_arr=se_arr,
                x_value=age,
                text=txt,
                where=where,
                fontsize=8,      # smaller font so labels don't overlap
                ypad_frac=0.03   # a bit of padding beyond the error bar
            )

# Axis labels: left side + bottom row
for r in range(4):
    axes[r, 0].set_ylabel("Cover (%)")

for c in range(3):
    axes[3, c].set_xlabel("Landscape age")

plt.tight_layout()

out = "F7_Species_Ageclass_12Panel.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {out}")

plt.show()
