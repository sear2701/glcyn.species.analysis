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
# Figure setup (3x4) to match example style
# -----------------------------
fig, axes = plt.subplots(
    nrows=4, ncols=3,  # 4 rows x 3 cols = 12 panels (portrait-ish like your example)
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

    m, se = mean_and_se(var)

    ax.errorbar(
        x, m, yerr=se,
        **err_kw,
        **line_kw
    )

    ax.set_title(titles[i], fontsize=11, fontweight="bold", pad=4)
    #ax.grid(True, alpha=0.25)

# Axis labels: left side + bottom row (like your example)
for r in range(4):
    axes[r, 0].set_ylabel("Cover (%)")

for c in range(3):
    axes[3, c].set_xlabel("Landscape age")

plt.tight_layout()

out = "Species_Ageclass_12Panel.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {out}")

plt.show()
