import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
path = "/workspaces/glcyn.species.analysis/data/species.cover.csv"
df = pd.read_csv(path)

# Ensure sorted x for plotting
df = df.copy()
df["ageclassn"] = df["ageclassn"]
df = df.sort_values("ageclassn")

# -----------------------------
# Panel definitions (9 panels)
# -----------------------------
pairs = [
    ("solcan", "verame"),
    ("saltra", "artlud"),
    ("equlae", "junbal"),
    ("vuloct", "polmon"),
    ("sacrav", "phraus"),
    ("brotec", "brorub"),
    ("bacsalf", "bacsaln"),
    ("salexi", "tamram"),
    ("salgoo", "popfre"),
]

# -----------------------------
# Summarize: mean + 1 SE by ageclassn for all variables in pairs
# -----------------------------
vars_needed = sorted(set([v for pair in pairs for v in pair]))
agg = {v: ["mean", "std", "count"] for v in vars_needed}

summ = df.groupby("ageclassn").agg(agg)
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
# Figure setup (3x3)
# -----------------------------
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True)

line_kw = dict(color="black", linewidth=2)
style1 = dict(marker="o", markersize=7, linestyle="-")
style2 = dict(marker="s", markersize=7, linestyle="--")

# If you prefer species names capitalized in legends:
def pretty(name: str) -> str:
    return name  # keep as-is; edit if you want formatting

# -----------------------------
# Plot each panel
# -----------------------------
for i, (v1, v2) in enumerate(pairs):
    r, c = divmod(i, 3)
    ax = axes[r, c]

    m1, se1 = mean_and_se(v1)
    m2, se2 = mean_and_se(v2)

    ax.errorbar(
        x, m1, yerr=se1,
        capsize=0, **line_kw, **style1, label=pretty(v1)
    )
    ax.errorbar(
        x, m2, yerr=se2,
        capsize=0, **line_kw, **style2, label=pretty(v2)
    )

    # Match the general look of your example
    ax.legend(loc="center right", frameon=True)
    ax.set_ylabel("Mean value ± 1 SE")

# Bottom row x-labels (or use a single shared label below)
for ax in axes[-1, :]:
    ax.set_xlabel("Landscape age")

plt.tight_layout()

out = "Species_Ageclass_9Panel.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {out}")

plt.show()
