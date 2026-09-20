#!/usr/bin/env python3
"""
King County Food Establishment Inspection – Statistical Analysis
================================================================
Source : https://data.kingcounty.gov/d/r878-4sxa  (export.csv)
Key    : Each ROW is a violation record, not an inspection.
         An inspection with N violations produces N rows sharing
         the same Inspection_Serial_Num.

This script:
  Step 1 – Inspect columns, dtypes, missing values, constant-field check.
  Step 2 – Build row-level and inspection-level datasets.
  Step 3 – Five analyses on both datasets, side by side.
  Step 4 – Circularity check (score vs summed violation points).
  Step 5 – Figures (PNG), results_summary.csv, results_summary_long.csv,
           and results_summary_side_by_side.csv.

All numeric outputs are computed directly from the data.
"""

import warnings
import os
import sys
import io
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (pearsonr, spearmanr, kruskal,
                         mannwhitneyu, chi2_contingency)
import statsmodels.api as sm
from statsmodels.formula.api import logit
import matplotlib
matplotlib.use("Agg")                       # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Argument Parsing ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Reproduce and correct statistical analysis of King County Food Inspections."
)
parser.add_argument(
    "--input",
    default="data/export.csv",
    help="Path to raw export.csv dataset (default: data/export.csv)"
)
parser.add_argument(
    "--output",
    default="outputs",
    help="Directory to save figures and CSV outputs (default: outputs)"
)
args = parser.parse_args()

DATA_FILE = args.input
OUT_DIR = args.output
os.makedirs(OUT_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")

# Metric collector for summary tables
results_rows = []

def record(analysis, level, stat_name, value, n=None):
    results_rows.append(dict(analysis=analysis, level=level,
                             statistic=stat_name, value=value, n=n))

print("=" * 72)
print("KING COUNTY FOOD INSPECTION ANALYSIS")
print("=" * 72)
print(f"Input file : {DATA_FILE}")
print(f"Output dir : {OUT_DIR}")

# =====================================================================
# STEP 1 – Inspect
# =====================================================================
print("\n── STEP 1: Data inspection ─────────────────────────────────")
if not os.path.exists(DATA_FILE):
    sys.exit(f"Error: Dataset file not found at '{DATA_FILE}'. Please see data/README.md.")

df_raw = pd.read_csv(DATA_FILE, dtype=str)
print(f"Rows loaded          : {len(df_raw):,}")
print(f"Columns ({len(df_raw.columns)}):")
for c in df_raw.columns:
    print(f"  {c:35s} dtype={df_raw[c].dtype}  missing={df_raw[c].isna().sum():,}")

# ── Column mapping ───────────────────────────────────────────────────
COL_MAP = {
    "Name":                       "Name",
    "Program Identifier":         "Program_Identifier",
    "Inspection Date":            "Inspection_Date",
    "Classification":             "Classification",
    "Address":                    "Address",
    "City":                       "City",
    "Zip Code":                   "Zip_Code",
    "Inspection Type":            "Inspection_Type",
    "Inspection Score":           "Inspection_Score",
    "Inspection Result":          "Inspection_Result",
    "Inspection Closed Business": "Inspection_Closed_Business",
    "Seating Range":              "Seating_Range",
    "Risk Category":              "Risk_Category",
    "Violation Type":             "Violation_Type",
    "Violation Description":      "Violation_Description",
    "Violation Points":           "Violation_Points",
    "Grade":                      "Grade",
    "Parcel Number":              "Parcel_Number",
    "Business_ID":                "Business_ID",
    "Inspection_Serial_Num":      "Inspection_Serial_Num",
}
print("\nColumn mapping applied:")
for orig, new in COL_MAP.items():
    if orig != new:
        print(f"  '{orig}'  →  '{new}'")
df_raw.rename(columns=COL_MAP, inplace=True)

# Unique inspection count
n_inspections_raw = df_raw["Inspection_Serial_Num"].nunique()
print(f"\nUnique Inspection_Serial_Num : {n_inspections_raw:,}")

# ── Constant-field check ─────────────────────────────────────────────
EXPECTED_VARYING = ["Violation_Type", "Violation_Description", "Violation_Points"]

print("\nConstant-field check (within each Inspection_Serial_Num):")
actually_constant = []
not_constant = []
for col in df_raw.columns:
    if col == "Inspection_Serial_Num":
        continue
    nuniq = df_raw.groupby("Inspection_Serial_Num")[col].nunique()
    max_u = nuniq.max()
    if max_u <= 1:
        actually_constant.append(col)
        tag = "✓ constant"
    else:
        not_constant.append(col)
        tag = f"✗ varies (max {max_u} distinct values within one inspection)"
    print(f"  {col:35s}  {tag}")

# ── Type conversions ─────────────────────────────────────────────────
df = df_raw.copy()
df["Inspection_Score"] = pd.to_numeric(df["Inspection_Score"], errors="coerce")
df["Violation_Points"] = pd.to_numeric(df["Violation_Points"], errors="coerce")
df["Risk_Category"]    = pd.to_numeric(df["Risk_Category"], errors="coerce")
df["Closed"] = df["Inspection_Closed_Business"].map(
    lambda x: 1 if str(x).strip().upper() == "YES" else 0
)

# Encode Violation Type
df["is_red"]  = (df["Violation_Type"].str.upper() == "RED").astype(int)
df["is_blue"] = (df["Violation_Type"].str.upper() == "BLUE").astype(int)

# Encode Grade ordinally
grade_order = {"NEEDS IMPROVEMENT": 1, "ADEQUATE": 2, "OKAY": 2, 
               "GOOD": 3, "EXCELLENT": 4}
df["Grade_ord"] = df["Grade"].str.upper().str.strip().map(grade_order)

# Encode Inspection Result
result_vals = sorted(df["Inspection_Result"].dropna().unique())
print(f"\nInspection Result values: {result_vals}")
result_map = {v: i for i, v in enumerate(result_vals)}
df["Result_enc"] = df["Inspection_Result"].map(result_map)

print(f"\nAfter type conversion  : {len(df):,} rows")

# =====================================================================
# STEP 2 – Build datasets
# =====================================================================
print("\n── STEP 2: Build datasets ──────────────────────────────────")

# 2a Row-level dataset
df_row = df.copy()
print(f"(a) Row-level dataset  : {len(df_row):,} rows")

# 2b Inspection-level dataset
insp_constant_cols = [c for c in actually_constant if c not in EXPECTED_VARYING]
keep_cols = ["Inspection_Serial_Num"] + insp_constant_cols
insp_base = df.drop_duplicates(subset="Inspection_Serial_Num")[keep_cols].copy()
print(f"    Inspection-level base (unique inspections): {len(insp_base):,}")

# Aggregated violation fields
viol_agg = df.groupby("Inspection_Serial_Num").agg(
    n_violations        = ("Violation_Type", lambda x: x.notna().sum()),
    n_red               = ("is_red", "sum"),
    n_blue              = ("is_blue", "sum"),
    total_violation_pts = ("Violation_Points", "sum"),
).reset_index()

df_insp = insp_base.merge(viol_agg, on="Inspection_Serial_Num", how="left")
df_insp["has_red"] = (df_insp["n_red"] > 0).astype(int)

df_insp["Inspection_Score"] = pd.to_numeric(df_insp["Inspection_Score"], errors="coerce")
df_insp["Risk_Category"]    = pd.to_numeric(df_insp["Risk_Category"], errors="coerce")
df_insp["Closed"] = df_insp["Inspection_Closed_Business"].map(
    lambda x: 1 if str(x).strip().upper() == "YES" else 0
)
df_insp["Grade_ord"]  = df_insp["Grade"].str.upper().str.strip().map(grade_order)
df_insp["Result_enc"] = df_insp["Inspection_Result"].map(result_map)

print(f"(b) Inspection-level   : {len(df_insp):,} rows")
print(f"    n_violations range : {df_insp['n_violations'].min()} – {df_insp['n_violations'].max()}")
print(f"    n_red range        : {df_insp['n_red'].min()} – {df_insp['n_red'].max()}")
print(f"    total_violation_pts: {df_insp['total_violation_pts'].min()} – {df_insp['total_violation_pts'].max()}")

# =====================================================================
# STEP 3 – Five Analyses
# =====================================================================
print("\n── STEP 3: Analyses ────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────
# 3.1  Correlation matrices
# ─────────────────────────────────────────────────────────────────────
print("\n── 3.1 Correlation matrices ────────────────────────────────")

# Row-level numeric columns
row_num_cols = ["Inspection_Score", "Violation_Points", "Risk_Category",
                "Grade_ord", "Result_enc", "Closed", "is_red", "is_blue"]
df_row_num = df_row[row_num_cols].dropna()
print(f"  Row-level: {len(df_row_num):,} complete cases on {row_num_cols}")
pearson_row  = df_row_num.corr(method="pearson")
spearman_row = df_row_num.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(pearson_row,  annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title("Pearson – Row Level")
sns.heatmap(spearman_row, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title("Spearman – Row Level")
plt.suptitle("Correlation Matrix (Row-Level Dataset)", fontsize=14)
savefig("corr_row_level.png")

for i in range(len(row_num_cols)):
    for j in range(i+1, len(row_num_cols)):
        c1, c2 = row_num_cols[i], row_num_cols[j]
        record("3.1_correlation", "row", f"pearson_{c1}_vs_{c2}",
               round(pearson_row.loc[c1, c2], 4), n=len(df_row_num))
        record("3.1_correlation", "row", f"spearman_{c1}_vs_{c2}",
               round(spearman_row.loc[c1, c2], 4), n=len(df_row_num))

# Inspection-level numeric columns
insp_num_cols = ["Inspection_Score", "total_violation_pts", "n_violations",
                 "has_red", "Risk_Category", "Grade_ord", "Result_enc", "Closed"]
df_insp_num = df_insp[insp_num_cols].dropna()
print(f"  Inspection-level: {len(df_insp_num):,} complete cases on {insp_num_cols}")
pearson_insp  = df_insp_num.corr(method="pearson")
spearman_insp = df_insp_num.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(pearson_insp,  annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title("Pearson – Inspection Level")
sns.heatmap(spearman_insp, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title("Spearman – Inspection Level")
plt.suptitle("Correlation Matrix (Inspection-Level Dataset)", fontsize=14)
savefig("corr_inspection_level.png")

for i in range(len(insp_num_cols)):
    for j in range(i+1, len(insp_num_cols)):
        c1, c2 = insp_num_cols[i], insp_num_cols[j]
        record("3.1_correlation", "inspection", f"pearson_{c1}_vs_{c2}",
               round(pearson_insp.loc[c1, c2], 4), n=len(df_insp_num))
        record("3.1_correlation", "inspection", f"spearman_{c1}_vs_{c2}",
               round(spearman_insp.loc[c1, c2], 4), n=len(df_insp_num))

# ─────────────────────────────────────────────────────────────────────
# 3.2  Inspection Score by Risk Category
# ─────────────────────────────────────────────────────────────────────
print("\n── 3.2 Inspection Score by Risk Category ───────────────────")

for label, dset in [("row", df_row), ("inspection", df_insp)]:
    sub = dset.dropna(subset=["Inspection_Score", "Risk_Category"])
    groups = sub.groupby("Risk_Category")["Inspection_Score"]
    print(f"\n  [{label.upper()} level]  n={len(sub):,}")
    for name, grp in groups:
        med = grp.median()
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr = q3 - q1
        print(f"    Risk {name}: median={med}, IQR={iqr:.1f} (Q1={q1}, Q3={q3}), n={len(grp):,}")
        record("3.2_score_by_risk", label, f"risk_{name}_median", med, n=len(grp))
        record("3.2_score_by_risk", label, f"risk_{name}_IQR", round(iqr, 2), n=len(grp))

    samples = [grp.values for _, grp in groups]
    if len(samples) >= 2:
        stat, p = kruskal(*samples)
        print(f"    Kruskal-Wallis H={stat:.4f}, p={p:.4e}")
        record("3.2_score_by_risk", label, "kruskal_H", round(stat, 4), n=len(sub))
        record("3.2_score_by_risk", label, "kruskal_p", p, n=len(sub))

    fig, ax = plt.subplots(figsize=(8, 5))
    risk_cats = sorted(sub["Risk_Category"].dropna().unique())
    sns.boxplot(data=sub, x="Risk_Category", y="Inspection_Score",
                hue="Risk_Category", legend=False,
                order=risk_cats, ax=ax, palette="Set2")
    ax.set_title(f"Inspection Score by Risk Category ({label.capitalize()} Level)")
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Inspection Score")
    savefig(f"boxplot_score_risk_{label}.png")

# ─────────────────────────────────────────────────────────────────────
# 3.3  Inspection Result by violation type
# ─────────────────────────────────────────────────────────────────────
print("\n── 3.3 Inspection Result by Violation Type ─────────────────")

# Row level – count plot
sub_row = df_row.dropna(subset=["Inspection_Result", "Violation_Type"])
print(f"  [ROW level]  n={len(sub_row):,}")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=sub_row, x="Inspection_Result", hue="Violation_Type", ax=ax,
              order=sorted(sub_row["Inspection_Result"].unique()))
ax.set_title("Violation Type Counts by Inspection Result (Row Level)")
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
savefig("countplot_result_violation_row.png")

ct_row = pd.crosstab(sub_row["Violation_Type"], sub_row["Inspection_Result"])
print("  Row-level cross-tab:")
print(ct_row.to_string())
for vt in ct_row.index:
    for ir in ct_row.columns:
        record("3.3_result_violation", "row", f"count_{vt}_{ir}", int(ct_row.loc[vt, ir]),
               n=len(sub_row))

# Inspection level – contingency table has_red × Inspection_Result
sub_insp = df_insp.dropna(subset=["Inspection_Result", "has_red"])
print(f"\n  [INSPECTION level]  n={len(sub_insp):,}")
ct_insp = pd.crosstab(sub_insp["has_red"], sub_insp["Inspection_Result"])
print("  Contingency table (has_red × Inspection_Result):")
print(ct_insp.to_string())

chi2, p_chi2, dof, expected = chi2_contingency(ct_insp)
n_ct = ct_insp.values.sum()
k = min(ct_insp.shape)
cramers_v = np.sqrt(chi2 / (n_ct * (k - 1))) if (k - 1) > 0 else np.nan
print(f"  Chi-square = {chi2:.4f}, p = {p_chi2:.4e}, dof = {dof}")
print(f"  Cramér's V = {cramers_v:.4f}")
record("3.3_result_violation", "inspection", "chi2", round(chi2, 4), n=n_ct)
record("3.3_result_violation", "inspection", "chi2_p", p_chi2, n=n_ct)
record("3.3_result_violation", "inspection", "cramers_v", round(cramers_v, 4), n=n_ct)
record("3.3_result_violation", "inspection", "dof", dof, n=n_ct)

fig, ax = plt.subplots(figsize=(10, 5))
ct_insp.plot(kind="bar", ax=ax, colormap="Set2")
ax.set_title("has_red × Inspection Result (Inspection Level)")
ax.set_xlabel("has_red (0=no red violations, 1=has red)")
ax.set_ylabel("Count")
plt.tight_layout()
savefig("contingency_hasred_result_insp.png")

# ─────────────────────────────────────────────────────────────────────
# 3.4  Score distribution by Inspection Type (Routine vs Return)
# ─────────────────────────────────────────────────────────────────────
print("\n── 3.4 Score by Inspection Type (Routine vs Return) ───────")

for label, dset in [("row", df_row), ("inspection", df_insp)]:
    sub = dset.dropna(subset=["Inspection_Score", "Inspection_Type"]).copy()
    sub["type_group"] = np.where(
        sub["Inspection_Type"].str.contains("Routine", case=False, na=False), "Routine",
        np.where(sub["Inspection_Type"].str.contains("Return", case=False, na=False), "Return",
                 "Other"))
    routine = sub.loc[sub["type_group"] == "Routine", "Inspection_Score"]
    ret     = sub.loc[sub["type_group"] == "Return",  "Inspection_Score"]
    print(f"\n  [{label.upper()} level]  Routine n={len(routine):,},  Return n={len(ret):,}")

    if len(routine) > 1 and len(ret) > 1:
        # Welch t-test
        t_stat, p_t = stats.ttest_ind(routine, ret, equal_var=False)
        pooled_std = np.sqrt((routine.std()**2 + ret.std()**2) / 2)
        cohens_d = (routine.mean() - ret.mean()) / pooled_std if pooled_std > 0 else np.nan
        print(f"    Welch t = {t_stat:.4f}, p = {p_t:.4e}")
        print(f"    Cohen's d = {cohens_d:.4f}")
        record("3.4_type_score", label, "welch_t", round(t_stat, 4),
               n=len(routine)+len(ret))
        record("3.4_type_score", label, "welch_p", p_t, n=len(routine)+len(ret))
        record("3.4_type_score", label, "cohens_d", round(cohens_d, 4),
               n=len(routine)+len(ret))

        # Mann-Whitney U
        u_stat, p_u = mannwhitneyu(routine, ret, alternative="two-sided")
        n1, n2 = len(routine), len(ret)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_u:.4e}")
        print(f"    Rank-biserial r = {rank_biserial:.4f}")
        record("3.4_type_score", label, "mann_whitney_U", round(u_stat, 1),
               n=n1+n2)
        record("3.4_type_score", label, "mann_whitney_p", p_u, n=n1+n2)
        record("3.4_type_score", label, "rank_biserial_r", round(rank_biserial, 4),
               n=n1+n2)

        record("3.4_type_score", label, "routine_mean", round(routine.mean(), 4), n=n1)
        record("3.4_type_score", label, "return_mean", round(ret.mean(), 4), n=n2)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(routine) > 1:
        routine.plot.kde(ax=ax, label=f"Routine (n={len(routine):,})", bw_method=0.3)
    if len(ret) > 1:
        ret.plot.kde(ax=ax, label=f"Return (n={len(ret):,})", bw_method=0.3)
    ax.set_title(f"Score Distribution: Routine vs Return ({label.capitalize()} Level)")
    ax.set_xlabel("Inspection Score")
    ax.legend()
    ax.set_xlim(left=-5)
    savefig(f"kde_routine_return_{label}.png")

# ─────────────────────────────────────────────────────────────────────
# 3.5  Logistic regression of closure on Inspection Score
# ─────────────────────────────────────────────────────────────────────
print("\n── 3.5 Logistic Regression: Closure ~ Inspection Score ─────")

for label, dset in [("row", df_row), ("inspection", df_insp)]:
    sub = dset.dropna(subset=["Inspection_Score", "Closed"]).copy()
    sub = sub[sub["Closed"].isin([0, 1])]
    n_closed = int(sub["Closed"].sum())
    n_total  = len(sub)
    print(f"\n  [{label.upper()} level]  n={n_total:,},  closures={n_closed:,}")
    print(f"  NOTE: Closures are rare ({n_closed/n_total*100:.2f}%) and "
          f"high-score observations are sparse.")
    record("3.5_logistic", label, "n_total", n_total, n=n_total)
    record("3.5_logistic", label, "n_closures", n_closed, n=n_total)

    if n_closed < 5:
        print(f"  ⚠ Too few closures ({n_closed}) for reliable logistic regression – skipping.")
        record("3.5_logistic", label, "status", "skipped_too_few_events", n=n_total)
        continue

    try:
        sub_fit = sub[["Inspection_Score", "Closed"]].copy()
        sub_fit = sub_fit.rename(columns={"Inspection_Score": "score"})

        if label == "row":
            sub_fit["cluster"] = sub["Inspection_Serial_Num"].values
            model = logit("Closed ~ score", data=sub_fit).fit(
                disp=0, cov_type="cluster",
                cov_kwds={"groups": sub_fit["cluster"]})
        else:
            model = logit("Closed ~ score", data=sub_fit).fit(disp=0)

        coef = model.params["score"]
        ci   = model.conf_int().loc["score"]
        pval = model.pvalues["score"]
        print(f"    Coefficient (score): {coef:.6f}")
        print(f"    95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"    p-value: {pval:.4e}")
        if label == "row":
            print("    (Cluster-robust SEs on Inspection_Serial_Num)")

        record("3.5_logistic", label, "coef_score", round(coef, 6), n=n_total)
        record("3.5_logistic", label, "ci_lower", round(ci[0], 6), n=n_total)
        record("3.5_logistic", label, "ci_upper", round(ci[1], 6), n=n_total)
        record("3.5_logistic", label, "p_value", pval, n=n_total)

        score_min = sub_fit["score"].min()
        score_max = sub_fit["score"].max()
        score_grid = np.linspace(score_min, score_max, 300)
        pred_prob  = model.predict(pd.DataFrame({"score": score_grid}))

        fig, ax = plt.subplots(figsize=(8, 5))
        jitter = np.random.uniform(-0.02, 0.02, size=len(sub_fit))
        ax.scatter(sub_fit["score"], sub_fit["Closed"] + jitter,
                   alpha=0.05, s=5, color="grey", label="Observed (jittered)")
        ax.plot(score_grid, pred_prob, color="red", linewidth=2,
                label="Fitted logistic curve")
        ax.set_xlabel("Inspection Score")
        ax.set_ylabel("P(Closure)")
        ax.set_title(f"Logistic Regression: Closure ~ Score ({label.capitalize()} Level)\n"
                     f"n={n_total:,}, closures={n_closed:,} "
                     f"({'cluster-robust SE' if label=='row' else 'standard SE'})")
        ax.legend()
        ax.set_xlim(score_min - 2, score_max + 2)
        savefig(f"logistic_closure_{label}.png")

    except Exception as e:
        print(f"  ✗ Logistic regression failed: {e}")
        record("3.5_logistic", label, "status", f"failed: {e}", n=n_total)

# =====================================================================
# STEP 4 – Circularity check
# =====================================================================
print("\n── STEP 4: Circularity check (Score vs summed Violation Points) ─")

circ = df_insp[["Inspection_Serial_Num", "Inspection_Score",
                 "total_violation_pts"]].dropna()
r_p, p_p = pearsonr(circ["Inspection_Score"], circ["total_violation_pts"])
r_s, p_s = spearmanr(circ["Inspection_Score"], circ["total_violation_pts"])
n_circ = len(circ)

exact_match = (circ["Inspection_Score"] == circ["total_violation_pts"]).sum()
pct_exact = exact_match / n_circ * 100

print(f"  n = {n_circ:,}")
print(f"  Pearson  r = {r_p:.4f}, p = {p_p:.4e}")
print(f"  Spearman r = {r_s:.4f}, p = {p_s:.4e}")
print(f"  Exact match (score == sum_violation_pts): {exact_match:,} / {n_circ:,} ({pct_exact:.2f}%)")
if r_p > 0.9:
    print("  ⚠ CIRCULARITY: Inspection Score is very strongly determined by "
          "the sum of Violation Points. Correlations involving both score and "
          "violation points are NOT independent findings – one is essentially "
          "a function of the other.")
elif r_p > 0.7:
    print("  ⚠ HIGH OVERLAP: Score and summed violation points are highly "
          "correlated, suggesting substantial overlap in what they measure.")
else:
    print("  Score and summed violation points are moderately correlated.")

record("4_circularity", "inspection", "pearson_r", round(r_p, 4), n=n_circ)
record("4_circularity", "inspection", "pearson_p", p_p, n=n_circ)
record("4_circularity", "inspection", "spearman_r", round(r_s, 4), n=n_circ)
record("4_circularity", "inspection", "spearman_p", p_s, n=n_circ)
record("4_circularity", "inspection", "exact_match_count", exact_match, n=n_circ)
record("4_circularity", "inspection", "exact_match_pct", round(pct_exact, 2), n=n_circ)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(circ["total_violation_pts"], circ["Inspection_Score"],
           alpha=0.05, s=3, color="steelblue")
lims = [0, max(circ["total_violation_pts"].max(), circ["Inspection_Score"].max()) + 5]
ax.plot(lims, lims, "--", color="red", label="y = x (identity)")
ax.set_xlabel("Sum of Violation Points per Inspection")
ax.set_ylabel("Inspection Score")
# Title formatting: exact match rate with two decimal places (99.96%)
ax.set_title(f"Circularity Check: Score vs Summed Violation Points\n"
             f"Pearson r={r_p:.3f}, Exact match={pct_exact:.2f}%")
ax.legend()
savefig("circularity_score_vs_points.png")

# =====================================================================
# STEP 5 – Outputs
# =====================================================================
print("\n── STEP 5: Outputs ─────────────────────────────────────────")

# 5a. Save summary CSV tables
df_results_long = pd.DataFrame(results_rows)
long_path = os.path.join(OUT_DIR, "results_summary_long.csv")
df_results_long.to_csv(long_path, index=False)

row_df = df_results_long[df_results_long['level'] == 'row'].copy()
insp_df = df_results_long[df_results_long['level'] == 'inspection'].copy()

# Raw unaligned side-by-side
df_raw_sbs = pd.merge(
    row_df[['analysis', 'statistic', 'value', 'n']].rename(columns={'value': 'row_value', 'n': 'row_n'}),
    insp_df[['analysis', 'statistic', 'value', 'n']].rename(columns={'value': 'inspection_value', 'n': 'inspection_n'}),
    on=['analysis', 'statistic'],
    how='outer'
)
sbs_path = os.path.join(OUT_DIR, "results_summary_side_by_side.csv")
df_raw_sbs.to_csv(sbs_path, index=False)

# Aligned side-by-side summary
concept_map = {
    'pearson_Inspection_Score_vs_Violation_Points': 'pearson_Inspection_Score_vs_Violation_Points (pts / total_pts)',
    'pearson_Inspection_Score_vs_total_violation_pts': 'pearson_Inspection_Score_vs_Violation_Points (pts / total_pts)',
    'spearman_Inspection_Score_vs_Violation_Points': 'spearman_Inspection_Score_vs_Violation_Points (pts / total_pts)',
    'spearman_Inspection_Score_vs_total_violation_pts': 'spearman_Inspection_Score_vs_Violation_Points (pts / total_pts)',
    'pearson_Inspection_Score_vs_is_red': 'pearson_Inspection_Score_vs_Red_Violation (is_red / has_red)',
    'pearson_Inspection_Score_vs_has_red': 'pearson_Inspection_Score_vs_Red_Violation (is_red / has_red)',
    'spearman_Inspection_Score_vs_is_red': 'spearman_Inspection_Score_vs_Red_Violation (is_red / has_red)',
    'spearman_Inspection_Score_vs_has_red': 'spearman_Inspection_Score_vs_Red_Violation (is_red / has_red)',
}

row_df['stat_aligned'] = row_df['statistic'].replace(concept_map)
insp_df['stat_aligned'] = insp_df['statistic'].replace(concept_map)

df_side_by_side = pd.merge(
    row_df[['analysis', 'stat_aligned', 'statistic', 'value', 'n']].rename(
        columns={'value': 'row_value', 'n': 'row_n', 'statistic': 'row_stat_name'}),
    insp_df[['analysis', 'stat_aligned', 'statistic', 'value', 'n']].rename(
        columns={'value': 'inspection_value', 'n': 'inspection_n', 'statistic': 'insp_stat_name'}),
    on=['analysis', 'stat_aligned'],
    how='outer'
)

results_path = os.path.join(OUT_DIR, "results_summary.csv")
df_side_by_side.to_csv(results_path, index=False)

print(f"  results_summary.csv (side-by-side) saved ({len(df_side_by_side)} rows): {results_path}")
print(f"  results_summary_long.csv saved ({len(df_results_long)} rows): {long_path}")
print(f"  results_summary_side_by_side.csv saved ({len(df_raw_sbs)} rows): {sbs_path}")

print("\n" + "=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)
