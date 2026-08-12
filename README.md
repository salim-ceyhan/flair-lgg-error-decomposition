# FLAIR LGG Candidate-Selection Error Decomposition

This private review repository accompanies the manuscript **“Error Decomposition of Unsupervised FLAIR-Only LGG Candidate Selection”**. It contains the frozen candidate-construction implementation, evaluation programs, publication figures, and numerical result records needed to audit the reported analysis.

## Authors

- Salim Ceyhan — conceptualization, theoretical formulation, methodology, formal analysis, investigation, validation, and supervision ([ORCID](https://orcid.org/0000-0003-0274-6175))
- Haydar Kılıç — software, implementation, visualization, and original-draft writing ([ORCID](https://orcid.org/0000-0002-2551-3772))

Correspondence: salim.ceyhan@bilecik.edu.tr

## Repository scope

- `src/`: reusable Finsler/NewMetric and candidate-selection components.
- `stage1_finsler_test/`: candidate-generation dependencies used by the frozen study.
- `study/`: construction, evaluation, validation, analysis, export, and visualization programs.
- `results/pure_flair_p1_corrected_finsler/`: frozen result tables and audit artifacts supporting the manuscript.
- `manuscript/`: LaTeX source and canonical manuscript figures.
- `vendor/FAC-codes/`: upstream reference implementation retained for provenance.

No patient images, masks, credentials, or personally identifiable information are included.

## Data

The experiments use de-identified public or controlled-access imaging cohorts described in the manuscript: TCGA-LGG, UCSF-PDGM, and BraTS 2023. Data must be obtained from the original custodians under their applicable terms and are intentionally not redistributed here.

For the primary TCGA-LGG analysis, the expected derived layout is:

```text
data/tcga_lgg_dataset/<case_id>/flair.npy
data/tcga_lgg_dataset/<case_id>/mask.npy
```

Each array represents the selected 256 x 256 FLAIR slice and its binary reference mask. Dataset paths can be overridden through the command-line options exposed by the individual study programs.

## Environment

Use Python 3.10 or later and install the declared packages:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

The canonical NewMetric execution used for the reported experiments relies on the separately distributed FACSeg-Fast backend. Set `FACSEG_ROOT` to its repository root before recomputing candidate pools. The reference FAC implementation is included under `vendor/` for provenance, but it is not silently substituted for the frozen backend because doing so would change the computational experiment.

## Reproducing the analysis

Run commands from the repository root with the root and `study/` directories on `PYTHONPATH`. The main audit sequence is:

1. Build or inspect the frozen candidate pool with `study/core/build_frozen_candidate_pool.py`.
2. Reproduce the candidate-budget and component analyses with the programs under `study/evaluation/` and `study/analysis/`.
3. Recreate publication graphics using the corresponding programs under `study/visualization/`.
4. Compare generated outputs against the immutable records under `results/pure_flair_p1_corrected_finsler/`.

Every program exposes its accepted paths and options through `--help` where applicable. The method uses anisotropic NewMetric diffusion; Gaussian filtering is not part of the reported procedure.

## Review availability

This repository is private during peer review. Confidential access can be granted to editors and reviewers on request. Subject to journal and data-governance requirements, the repository is intended for public release upon article acceptance.

