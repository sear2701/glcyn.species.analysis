import pandas as pd
import numpy as np
from scipy import stats

# -----------------------------
# USER SETTINGS
# -----------------------------
DATA_PATH = "/workspaces/glcyn.species.analysis/data/spp.cover.sites.top25.csv"
GROUP_COL = "ab"  # sorting variable with values "above"/"below"
OUT_CSV = "/workspaces/glcyn.species.analysis/data/last35_above_below_mean_tests.csv"

ALPHA = 0.05

# -----------------------------
# HELPERS
# -----------------------------
def normality_pvalue(x: np.ndarray):
    """
    Returns (test_name, pvalue) for a normality test on x.
    Strategy:
      - If n < 3: not enough data
      - If 3 <= n <= 5000: Shapiro-Wilk (recommended range)
      - If n > 5000: D’Agostino-Pearson (normaltest) if n>=8, else fallback
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 3:
        return ("normality_not_enough_data", np.nan)

    if n <= 5000:
        try:
            p = stats.shapiro(x).pvalue
            return ("shapiro", float(p))
        except Exception:
            return ("shapiro_failed", np.nan)

    # n > 5000
    if n >= 8:
        try:
            p = stats.normaltest(x).pvalue
            return ("dagostino_pearson", float(p))
        except Exception:
            return ("normaltest_failed", np.nan)

    return ("normality_not_enough_data", np.nan)

def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values in original order (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    pv = p[mask]
    n = pv.size
    if n == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev

    adj = np.clip(adj, 0, 1)
    tmp = np.empty(n, dtype=float)
    tmp[order] = adj
    out[mask] = tmp
    return out

# -----------------------------
# LOAD + CLEAN
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Standardize group labels
df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip().str.lower()

# Keep only "above"/"below"
df = df[df[GROUP_COL].isin(["above", "below"])].copy()

# Identify the last 35 columns
last35_cols = list(df.columns[-35:])

# Coerce last 35 to numeric where possible
for c in last35_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

above_df = df[df[GROUP_COL] == "above"]
below_df = df[df[GROUP_COL] == "below"]

print(f"Loaded: {DATA_PATH}")
print(f"Rows total (above/below only): {len(df):,}")
print(f"Rows above: {len(above_df):,} | below: {len(below_df):,}")
print(f"Analyzing last 35 columns: {last35_cols[0]} ... {last35_cols[-1]}\n")

# -----------------------------
# ANALYZE EACH COLUMN
# -----------------------------
results = []
pvals = []

for col in last35_cols:
    x = above_df[col].to_numpy(dtype=float)
    y = below_df[col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n_above, n_below = x.size, y.size

    mean_above = float(np.mean(x)) if n_above > 0 else np.nan
    mean_below = float(np.mean(y)) if n_below > 0 else np.nan
    diff = mean_above - mean_below if np.isfinite(mean_above) and np.isfinite(mean_below) else np.nan

    # Normality tests per group
    norm_test_a, norm_p_a = normality_pvalue(x)
    norm_test_b, norm_p_b = normality_pvalue(y)

    # Decide test
    # Rule:
    #   - If both groups appear normal (p >= alpha) AND both have n>=2 -> Welch t-test
    #   - Else -> Mann–Whitney U (nonparametric, compares distributions; often used for median shift)
    test_used = None
    stat = np.nan
    pval = np.nan

    enough_for_t = (n_above >= 2 and n_below >= 2)
    enough_for_u = (n_above >= 1 and n_below >= 1)

    both_normal = (np.isfinite(norm_p_a) and np.isfinite(norm_p_b) and (norm_p_a >= ALPHA) and (norm_p_b >= ALPHA))

    if enough_for_t and both_normal:
        # Welch's t-test (robust to unequal variances)
        test_used = "welch_ttest"
        tt = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        stat = float(tt.statistic)
        pval = float(tt.pvalue)
    elif enough_for_u:
        # Mann–Whitney U (two-sided)
        # Note: requires at least 1 observation per group
        test_used = "mann_whitney_u"
        try:
            mu = stats.mannwhitneyu(x, y, alternative="two-sided")
            stat = float(mu.statistic)
            pval = float(mu.pvalue)
        except Exception:
            test_used = "mann_whitney_u_failed"
            stat = np.nan
            pval = np.nan
    else:
        test_used = "not_enough_data"

    results.append({
        "column": col,
        "n_above": int(n_above),
        "mean_above": mean_above,
        "n_below": int(n_below),
        "mean_below": mean_below,
        "mean_diff_above_minus_below": diff,
        "normality_test_above": norm_test_a,
        "normality_p_above": norm_p_a,
        "normality_test_below": norm_test_b,
        "normality_p_below": norm_p_b,
        "test_used": test_used,
        "test_statistic": stat,
        "p_value": pval,
    })
    pvals.append(pval)

res = pd.DataFrame(results)

# Multiple-testing adjustment across 35 columns (BH-FDR)
res["p_value_bh_fdr"] = benjamini_hochberg(res["p_value"].values)

# Sort for printing
res_print = res.sort_values(["p_value", "column"], na_position="last")

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("=== Above vs Below: mean comparisons for last 35 columns ===")
with pd.option_context("display.max_rows", 200, "display.width", 180):
    print(
        res_print[[
            "column",
            "n_above", "mean_above",
            "n_below", "mean_below",
            "mean_diff_above_minus_below",
            "normality_test_above", "normality_p_above",
            "normality_test_below", "normality_p_below",
            "test_used", "test_statistic",
            "p_value", "p_value_bh_fdr"
        ]].to_string(index=False)
    )

# -----------------------------
# WRITE CSV
# -----------------------------
res.to_csv(OUT_CSV, index=False)
print(f"\nWrote results CSV to: {OUT_CSV}")
