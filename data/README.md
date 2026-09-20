# Dataset Download & Verification

This directory is intended to store the King County Food Establishment Inspection dataset (`export.csv`). The CSV file itself is ignored by Git and must be downloaded separately.

## Download Instructions

1. Visit the King County Open Data portal:
   - **URL**: [https://data.kingcounty.gov/d/r878-4sxa](https://data.kingcounty.gov/d/r878-4sxa)
2. Click **Export** &rarr; **CSV** to download the dataset.
3. Save the downloaded file as `data/export.csv`.

## Reference File Provenance

The analysis in this repository was executed on the following snapshot:

- **Source URL**: `https://data.kingcounty.gov/d/r878-4sxa`
- **Download Date**: `2026-08-12`
- **File Name**: `export.csv`
- **File Size**: `30,763,986` bytes
- **SHA-256 Checksum**:
  ```
  f4c0ec767cd41071d83b971fbd1444117a0bc514666f581013370d67400c3a2d
  ```

To verify the integrity of your downloaded file on Windows PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 data/export.csv
```
On Linux/macOS:
```bash
sha256sum data/export.csv
```

> **Note**: King County updates this dataset over time as new inspections occur. Re-downloads from later dates will reflect updated records and different total counts.
