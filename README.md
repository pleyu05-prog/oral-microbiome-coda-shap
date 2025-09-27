\# Oral Microbiome (HMP) — Oral Subsite Classification (CoDA + SHAP)



\## What

Classify \*\*oral subsites\*\* (tongue/buccal/plaque/saliva/…) from 16S profiles using \*\*compositional data analysis (CLR)\*\* and ML (L1-LogReg \& XGBoost). Reproducible scripts + figures included.



\## Data

\- Kaggle: `antaresnyc/human-metagenomics` (contains `abundance.csv`)

\- We filter \*\*oral sites\*\* and apply \*\*relative abundance → CLR\*\* transform.



\## Reproduce

```bash

\# prepare (generate features\_clr.csv \& labels.csv)

python scripts/prepare\_hmp\_oral.py

\# train + explain (save confusion matrices \& SHAP bar plots)

python scripts/train\_microbiome\_clf.py



\## Results (this run)

LogRegL1  acc=0.932  macroF1=0.522

XGBoost   acc=0.922  macroF1=0.500



\## Figures

!\[LogReg CM](assets/LogRegL1\_confusion.png)

!\[XGB CM](assets/XGBoost\_confusion.png)

!\[SHAP Linear](assets/shap\_top20\_linear.png)

!\[SHAP XGB](assets/shap\_top20\_xgb.png)



\## Notes

\- Pipeline: feature filter (≥1% present) → relative abundance → \*\*CLR\*\* → models (LogReg L1 / XGBoost).

\- Explainability: \*\*SHAP\*\* top-20 features for both models; map taxa to (e)HOMD for biological insight.

\- Splits: 80/20 stratified; set `random\_state=42` for reproducibility.

\- Caveats: public metagenomics mix datasets/batches; results reflect subsite separability, not clinical diagnosis.

\- Disclaimer: research/education only, \*\*not\*\* for clinical use.









