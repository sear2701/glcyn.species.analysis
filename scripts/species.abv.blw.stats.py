import pandas as pd
import numpy as np

from scipy import stats

# ---------------------------
# CONFIG
# ---------------------------
INPUT_CSV = "/workspaces/glcyn.species.analysis/data/species.cover.site.csv"
OUTPUT_CSV = "ab_group_comparison_last36.csv"

GROUP_COL = "ab"          # grouping variable
ABOVE_LABEL = "above"     # value in ab column for "above"
BELOW_LABEL = "below"     # value in ab column for "below"

ALPHA_NORMALITY = 0.05    # threshold for normality decision
# ---------------------------


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, forcing non-numeric to NaN."""
    return pd.to_numeric(series, errors="coerce")


def normality_test_p(x: np.ndarray) -> float:
    """
    Return a p-value for normality test.
    - If n < 3: not enough data -> NaN
    - If 3 <= n <= 5000: Shapiro-Wilk (common choice)
    - If n > 5000: D’Agostino-Pearson normaltest (SciPy recommends Shapiro not for huge n)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size

    if n < 3:
        return np.nan

    if n <= 5000:
        # Shapiro-Wilk
        try:
            return float(stats.shapiro(x).pvalue)
        except Exception:
            return np.nan
    else:
        # normaltest requires n >= 8
        if n < 8:
            return np.nan
        try:
            return float(stats.normaltest(x).pvalue)
        except Exception:
            return np.nan


def pick_variable_columns(df: pd.DataFrame) -> list[str]:
    """
    Prefer selecting columns from ACENIG..VULOCT (inclusive) if present and the slice is 36 cols.
    Otherwise fall back to the last 36 columns.
    """
    cols = list(df.columns)

    if "ACENIG" in cols and "VULOCT" in cols:
        i0 = cols.index("ACENIG")
        i1 = cols.index("VULOCT")
        if i0 <= i1:
            slice_cols = cols[i0 : i1 + 1]
            if len(slice_cols) == 36:
                return slice_cols

    # fallback: last 36 columns
    return cols[-36:]


def main():
    df = pd.read_csv(INPUT_CSV)

    if GROUP_COL not in df.columns:
        raise ValueError(f"Grouping column '{GROUP_COL}' not found in the dataset.")

    var_cols = pick_variable_columns(df)

    # Make sure we don't accidentally include the grouping column
    var_cols = [c for c in var_cols if c != GROUP_COL]

    # Subset by groups (case-insensitive safety)
    ab_series = df[GROUP_COL].astype(str).str.strip().str.lower()
    above_mask = ab_series == str(ABOVE_LABEL).lower()
    below_mask = ab_series == str(BELOW_LABEL).lower()

    if above_mask.sum() == 0 or below_mask.sum() == 0:
        raise ValueError(
            f"Could not find rows for both groups. Counts -> "
            f"{ABOVE_LABEL}: {above_mask.sum()}, {BELOW_LABEL}: {below_mask.sum()}"
        )

    results = []

    print("\n=== Group comparison: 'above' vs 'below' for selected 36 variables ===\n")

    for col in var_cols:
        x_above = coerce_numeric(df.loc[above_mask, col]).to_numpy()
        x_below = coerce_numeric(df.loc[below_mask, col]).to_numpy()

        # Drop NaNs for analysis
        xa = x_above[~np.isnan(x_above)]
        xb = x_below[~np.isnan(x_below)]

        n_above = xa.size
        n_below = xb.size

        mean_above = float(np.nanmean(xa)) if n_above > 0 else np.nan
        mean_below = float(np.nanmean(xb)) if n_below > 0 else np.nan

        # Normality p-values (per group)
        pnorm_above = normality_test_p(xa)
        pnorm_below = normality_test_p(xb)

        above_normal = (not np.isnan(pnorm_above)) and (pnorm_above > ALPHA_NORMALITY)
        below_normal = (not np.isnan(pnorm_below)) and (pnorm_below > ALPHA_NORMALITY)

        # Choose test
        test_name = None
        stat = np.nan
        pval = np.nan

        # Need enough data to test
        if n_above >= 2 and n_below >= 2:
            if above_normal and below_normal:
                # Use Welch's t-test (robust to unequal variances)
                test_name = "Welch_ttest"
                t = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
                stat = float(t.statistic)
                pval = float(t.pvalue)
            else:
                # Nonparametric Mann-Whitney U
                # Use two-sided; works for independent samples
                test_name = "MannWhitneyU"
                try:
                    u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                    stat = float(u.statistic)
                    pval = float(u.pvalue)
                except ValueError:
                    # e.g., all values identical can sometimes cause issues in older SciPy
                    test_name = "MannWhitneyU_failed"
                    stat = np.nan
                    pval = np.nan
        else:
            test_name = "Insufficient_n"

        diff = mean_above - mean_below if (not np.isnan(mean_above) and not np.isnan(mean_below)) else np.nan

        results.append(
            {
                "variable": col,
                "n_above": n_above,
                "n_below": n_below,
                "mean_above": mean_above,
                "mean_below": mean_below,
                "mean_diff_above_minus_below": diff,
                "normality_p_above": pnorm_above,
                "normality_p_below": pnorm_below,
                "test_used": test_name,
                "test_statistic": stat,
                "p_value": pval,
                "alpha_normality": ALPHA_NORMALITY,
            }
        )

        # Print a compact line to the command window
        print(
            f"{col:>10} | nA={n_above:4d} meanA={mean_above: .6g} | "
            f"nB={n_below:4d} meanB={mean_below: .6g} | "
            f"diff={diff: .6g} | "
            f"pNorm(A)={pnorm_above if not np.isnan(pnorm_above) else np.nan: .3g} "
            f"pNorm(B)={pnorm_below if not np.isnan(pnorm_below) else np.nan: .3g} | "
            f"{test_name} p={pval if not np.isnan(pval) else np.nan: .3g}"
        )

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
