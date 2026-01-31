import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# ----------------------------
# Load data
# ----------------------------
path = "/workspaces/glcyn.species.analysis/data/species.cover.csv"
df = pd.read_csv(path)

# Use ageclassn as the grouping variable; also create a string/categorical version
if "ageclassn" not in df.columns:
    raise ValueError("Expected column 'ageclassn' not found in the CSV.")

df["ageclassn_cat"] = df["ageclassn"].astype(str)

# Variables to test
variables = [
    "brotec", "brorub", "polmon", "phraus", "sacrav", "saltra",
    "bacsalf", "bacsaln", "salexi", "salgoo", "tamram", "popfre"
]

# ----------------------------
# Helpers
# ----------------------------
def shapiro_p(x: np.ndarray):
    """Shapiro-Wilk normality p-value; returns NaN if not applicable."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or len(x) > 5000:
        return np.nan
    try:
        return float(stats.shapiro(x).pvalue)
    except Exception:
        return np.nan

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; preserves NaNs."""
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    if mask.sum() == 0:
        return q
    _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
    q[mask] = qvals
    return q

def pairwise_mwu_bh(sub: pd.DataFrame, group_col: str, value_col: str):
    """
    Nonparametric posthoc: pairwise Mann-Whitney U with BH-FDR across pairwise tests.
    Returns a DataFrame.
    """
    groups = sorted(sub[group_col].dropna().unique(), key=lambda z: float(z) if str(z).replace(".","",1).isdigit() else str(z))
    vals = {g: sub.loc[sub[group_col] == g, value_col].dropna().to_numpy(dtype=float) for g in groups}

    rows = []
    pvals = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x1, x2 = vals[g1], vals[g2]
            if len(x1) < 2 or len(x2) < 2:
                u_stat, p = np.nan, np.nan
            else:
                u = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                u_stat, p = float(u.statistic), float(u.pvalue)
            rows.append([g1, g2, u_stat, p])
            pvals.append(p)

    qvals = bh_fdr(pvals)
    out = pd.DataFrame(rows, columns=["group1", "group2", "mw_u_stat", "p_value"])
    out["q_value_BH_FDR"] = qvals
    return out

# Sort age classes numerically when possible for stable output
def sort_age_levels(levels):
    def keyfun(z):
        s = str(z)
        try:
            return (0, float(s))
        except Exception:
            return (1, s)
    return sorted(levels, key=keyfun)

age_levels = sort_age_levels(df["ageclassn_cat"].dropna().unique())

# ----------------------------
# Run tests per variable
# ----------------------------
overall_rows = []
posthoc_tables = []  # will be concatenated with a "variable" column

alpha_posthoc = 0.10  # run posthoc when overall p < 0.1 as requested

for var in variables:
    if var not in df.columns:
        overall_rows.append({
            "variable": var,
            "test_used": "MISSING COLUMN",
            "statistic": np.nan,
            "p_value": np.nan,
            "levene_p": np.nan,
            "normality_note": np.nan,
            "min_group_n": np.nan
        })
        continue

    sub = df[["ageclassn_cat", var]].dropna().copy()
    sub = sub[sub["ageclassn_cat"].isin(age_levels)]

    # Build group arrays
    group_arrays = []
    group_ns = []
    shapiro_ps = []
    for g in age_levels:
        arr = sub.loc[sub["ageclassn_cat"] == g, var].to_numpy(dtype=float)
        if len(arr) > 0:
            group_arrays.append(arr)
        group_ns.append(len(arr))
        shapiro_ps.append(shapiro_p(arr))

    nonempty_groups = sum(n > 0 for n in group_ns)
    if nonempty_groups < 2:
        overall_rows.append({
            "variable": var,
            "test_used": "INSUFFICIENT GROUPS",
            "statistic": np.nan,
            "p_value": np.nan,
            "levene_p": np.nan,
            "normality_note": "Not enough age classes with data",
            "min_group_n": int(np.min([n for n in group_ns if n > 0])) if any(n > 0 for n in group_ns) else np.nan
        })
        continue

    # Normality rule:
    # If we can't run Shapiro for groups (small n), default to nonparametric.
    shapiro_valid = [p for p in shapiro_ps if not np.isnan(p)]
    any_non_normal = any(p <= 0.05 for p in shapiro_valid) if shapiro_valid else True

    # Levene variance homogeneity (only meaningful if enough samples)
    levene_p = np.nan
    try:
        arrays_for_levene = [a for a in group_arrays if len(a) >= 2]
        if len(arrays_for_levene) >= 2:
            levene_p = float(stats.levene(*arrays_for_levene, center="median").pvalue)
    except Exception:
        levene_p = np.nan

    # Choose ANOVA if normal + homogeneous variance; otherwise Kruskal-Wallis
    use_anova = (not any_non_normal) and (not np.isnan(levene_p) and levene_p > 0.05)

    stat = np.nan
    pval = np.nan
    test_used = ""

    if use_anova:
        tmp = sub.rename(columns={var: "y"}).copy()
        model = smf.ols("y ~ C(ageclassn_cat)", data=tmp).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        stat = float(anova_tbl.loc["C(ageclassn_cat)", "F"])
        pval = float(anova_tbl.loc["C(ageclassn_cat)", "PR(>F)"])
        test_used = "ANOVA (1-way)"

        # Posthoc when p < 0.1
        if np.isfinite(pval) and pval < alpha_posthoc:
            tuk = pairwise_tukeyhsd(endog=tmp["y"], groups=tmp["ageclassn_cat"], alpha=alpha_posthoc)
            ph = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
            ph.insert(0, "variable", var)
            ph.insert(1, "posthoc_test", "Tukey HSD")
            posthoc_tables.append(ph)

    else:
        # Kruskal-Wallis across age classes
        kw = stats.kruskal(*group_arrays, nan_policy="omit")
        stat = float(kw.statistic)
        pval = float(kw.pvalue)
        test_used = "Kruskal-Wallis"

        # Posthoc when p < 0.1
        if np.isfinite(pval) and pval < alpha_posthoc:
            ph = pairwise_mwu_bh(sub, "ageclassn_cat", var)
            ph.insert(0, "variable", var)
            ph.insert(1, "posthoc_test", "Pairwise MWU + BH-FDR")
            posthoc_tables.append(ph)

    overall_rows.append({
        "variable": var,
        "test_used": test_used,
        "statistic": stat,
        "p_value": pval,
        "levene_p": levene_p,
        "normality_note": (
            "All Shapiro p>0.05 (where run) and Levene p>0.05" if use_anova
            else "Non-normal group(s), unequal variance, or insufficient n for Shapiro → nonparametric"
        ),
        "min_group_n": int(np.min([n for n in group_ns if n > 0])) if any(n > 0 for n in group_ns) else np.nan
    })

overall_df = pd.DataFrame(overall_rows)

# Optional: FDR across variables for the overall tests (often useful)
overall_df["q_value_BH_FDR_overall"] = bh_fdr(overall_df["p_value"].to_numpy())

posthoc_df = pd.concat(posthoc_tables, ignore_index=True) if posthoc_tables else pd.DataFrame()

# ----------------------------
# Print to command window
# ----------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

print("\n" + "=" * 90)
print("OVERALL TESTS: Differences among age classes (ageclassn)")
print("=" * 90)
print(overall_df[[
    "variable", "test_used", "statistic", "p_value", "q_value_BH_FDR_overall",
    "levene_p", "min_group_n", "normality_note"
]].to_string(index=False))

if not posthoc_df.empty:
    print("\n" + "=" * 90)
    print(f"POSTHOC TESTS (performed when overall p < {alpha_posthoc})")
    print("=" * 90)
    print(posthoc_df.to_string(index=False))
else:
    print(f"\nNo posthoc tests run (no overall p < {alpha_posthoc}).")

# ----------------------------
# Save to CSV files
# ----------------------------
overall_out = "ageclassn_species_overall_tests.csv"
overall_df.to_csv(overall_out, index=False)
print(f"\nSaved overall results to: {overall_out}")

posthoc_out = "ageclassn_species_posthoc_tests.csv"
posthoc_df.to_csv(posthoc_out, index=False)
print(f"Saved posthoc results to: {posthoc_out}")
