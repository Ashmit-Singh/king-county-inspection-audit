# King County Food Establishment Inspection: Statistical Reproduction and Correction

## 1. What This Is
This repository contains an inspection-level reproduction and correction of the King County, Washington Food Establishment Inspection statistical analysis. It serves as an empirical proxy dataset for a forthcoming Zero-Knowledge Machine Learning (ZK-ML) audit paper (in progress).

> **Important Notice**: This repository contains **no zero-knowledge proofs, circuit implementations, or cryptographic code**, and the planned ZK audit API is **not built yet**. This repository strictly provides the statistical baseline, data pipeline, and econometric corrections for the underlying audit dataset.

---

## 2. Data Source & Provenance
- **Data Source**: King County Open Data — Food Establishment Inspection Data
- **Portal URL**: [https://data.kingcounty.gov/d/r878-4sxa](https://data.kingcounty.gov/d/r878-4sxa)
- **Reference Download Date**: `2026-08-12`
- **File Name**: `export.csv` (size: 30,763,986 bytes)
- **SHA-256 Hash**:
  ```
  f4c0ec767cd41071d83b971fbd1444117a0bc514666f581013370d67400c3a2d
  ```

> **Temporal Variance Note**: The King County Department of Public Health continuously updates this dataset as new food establishment inspections occur. Downloading a fresh copy from the portal on subsequent dates will yield updated record counts and slightly altered summary statistics.

---

## 3. Key Findings: Row Level vs. Inspection Level

In the raw dataset, **each row represents an individual violation record, not an inspection**. When an establishment receives multiple violations during a single inspection, that inspection appears as multiple rows sharing the same `Inspection_Serial_Num`. Constant inspection-level fields (`Inspection Score`, `Inspection Result`, `Inspection Closed Business`, `Risk Category`, `Grade`, `Inspection Type`) are duplicated across those rows.

Analyzing this dataset at the uncorrected **row level** results in severe **pseudoreplication**—restaurants with more violations are duplicated and overweighted, distorting medians, biasing hypothesis tests, and masking the true data-generating process.

### Comparison Table

| Metric / Analysis | Row-Level Analysis (Uncorrected) | Inspection-Level Analysis (Corrected) | Inferential Impact |
| :--- | :---: | :---: | :--- |
| **Total Observations ($N$)** | $106,257$ | $71,649$ | Pseudoreplication deflates standard errors by $48\%$. |
| **Recorded Business Closures** | $151$ | **$28$** | Closures are extreme rare events ($0.039\%$), not $0.14\%$. |
| **Risk 3 Score Median (IQR)** | $10.0$ ($\text{IQR} = 25.0$) | **$0.0$** ($\text{IQR} = 10.0$) | Across all risk tiers (1, 2, and 3), median score is $0.0$. |
| **Routine vs Return: Welch $t$-test** | $t = 2.7481$, **$p = 0.0060$** | $t = 1.4361$, **$p = 0.1511$** | **Spurious finding corrected**: Mean scores do not differ significantly. |
| **Routine vs Return: Mann-Whitney $U$** | $U = 1.55 \times 10^8$, $p = 1.05 \times 10^{-24}$ | $U = 8.08 \times 10^7$, $p = 2.37 \times 10^{-34}$ | Non-parametric rank shift ($r_{\text{rb}} = -0.14$). |
| **Score vs Summed Violation Points** | Pearson $r = 0.5282$ | Pearson **$r = 0.9999$** | **Circularity**: Exact equality in **$71,622$ of $71,649$ ($99.96\%$)**. |
| **has_red $\times$ Result Association** | N/A (row-level count plot) | $\chi^2 = 70,631.34$, **Cramér's $V = 0.9929$** | Red violations deterministically predict Unsatisfactory status ($99.4\%$). |

---

## 4. How to Run

### Prerequisites
- Python 3.10 or 3.11
- Virtual environment tool (`venv` or `conda`)

### Step-by-Step Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ashmit-Singh/ZKML.git
   cd ZKML
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Acquire the dataset**:
   Download `export.csv` from [King County Open Data](https://data.kingcounty.gov/d/r878-4sxa) and place it into `data/export.csv` (see [`data/README.md`](data/README.md) for verification details).

5. **Execute the analysis**:
   ```bash
   python analysis/kc_food_inspection_analysis.py --input data/export.csv --output outputs
   ```

All figures and CSV summaries will be saved directly to `outputs/`.

---

## 5. Known Limitations & Methodological Notes

1. **Complete-Case Analysis for Correlations**:
   - Correlation matrices (Analysis 3.1) are computed on complete cases across the numeric feature subset.
   - For the inspection-level correlation, this restricts the sample to **$59,893$ of $71,649$ inspections** ($83.6\%$), primarily because $11,756$ inspections had missing values in `Risk Category` or `Grade`.
2. **Rare-Event Sparsity in Closure Regression**:
   - The logistic regression of business closure on score is based on only **$28$ total closure events** across $71,649$ inspections.
   - Observations with high inspection scores ($>80$) are extremely sparse. While higher scores are positively associated with closure ($\beta = 0.0564, p = 1.07 \times 10^{-43}$), standard asymptotic properties can be sensitive to rare events, and the model should not be extrapolated outside the observed domain.
3. **Establishment-Level Grouping (Unclustered by `Business_ID`)**:
   - The current inspection-level model aggregates to `Inspection_Serial_Num`, but does not yet apply multi-level clustering or random effects by `Business_ID` across longitudinal inspections of the same venue over time.
4. **Encoding Definitions**:
   - `Result_enc`: Alphabetical label encoding of `Inspection Result`:
     - `Complete` &rarr; $0$
     - `Satisfactory` &rarr; $1$
     - `Unsatisfactory` &rarr; $2$
   - `Grade_ord`: Ordinal mapping reflecting King County's food safety tier system:
     - `NEEDS IMPROVEMENT` &rarr; $1$
     - `ADEQUATE` / `OKAY` &rarr; $2$
     - `GOOD` &rarr; $3$
     - `EXCELLENT` &rarr; $4$

---

## 6. AI Assistance & Authorship Attribution
The analytical specification, hypotheses, statistical correction criteria, and verification standards were directed, audited, and checked by **Ashmit Singh**. The implementation script and plotting routines were drafted with AI programming assistance.
