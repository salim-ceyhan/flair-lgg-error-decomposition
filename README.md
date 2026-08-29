# FLAIR LGG Candidate-Selection Error Decomposition

This public research-software repository accompanies the manuscript **“Candidate Generation and Selection Bottlenecks in Training-Free Single-Slice FLAIR Lower-Grade Glioma Segmentation”**. It contains the frozen candidate-construction implementation, evaluation programs, and numerical result records needed to audit the reported analysis.

## Authors

- Salim Ceyhan — conceptualization, theoretical formulation, methodology, formal analysis, investigation, validation, and supervision ([ORCID](https://orcid.org/0000-0003-0274-6175))
- Haydar Kılıç — software, implementation, visualization, and original-draft writing ([ORCID](https://orcid.org/0000-0002-2551-3772))

Correspondence: salim.ceyhan@bilecik.edu.tr

## Repository scope

> **Canonical release boundary.** This repository supports only the
> `pure_flair_p1_corrected_finsler` manuscript analysis. Earlier ROI-gated,
> exploratory, or development pipelines in the parent research workspace are
> legacy evidence and must not be used to reproduce manuscript claims.

- `src/`: reusable Finsler/NewMetric and candidate-selection components.
- `facseg/`: bundled evaluation utilities required by the study programs.
- `stage1_finsler_test/`: candidate-generation dependencies used by the frozen study.
- `study/`: construction, evaluation, validation, analysis, export, and visualization programs.
- `results/pure_flair_p1_corrected_finsler/`: frozen result tables and audit artifacts supporting the manuscript.

No patient images, masks, qualitative medical-image panels, credentials, or personally identifiable information are included. Publication figures containing medical images are deliberately excluded; the corresponding generation programs remain available under `study/visualization/`.

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

For an exact recreation of the verified publication workstation, use the
fully pinned lock file instead:

```bash
.venv/Scripts/python -m pip install -r requirements-lock.txt
```

The lock and `.python-version` record Python 3.14.4 and the exact package
versions used for the final integrity audit. `requirements.txt` remains the
portable compatibility range for Python 3.10 or later.

The canonical execution uses the theory-aligned local NewMetric implementation in
`src/newmetric_corrected.py` with beta 5.0, time step 0.15, and three
iterations. No separately distributed FACSeg-Fast checkout or `FACSEG_ROOT`
setting is required. The proprietary upstream FAC reference package is not
redistributed in this release.

## Reproducing the analysis

Run commands from the repository root with the root and `study/` directories on `PYTHONPATH`. The main audit sequence is:

1. Build or inspect the frozen candidate pool with `study/core/build_frozen_candidate_pool.py`.
2. Reproduce the candidate-budget and component analyses with the programs under `study/evaluation/` and `study/analysis/`.
3. Recreate publication graphics using the corresponding programs under `study/visualization/`.
4. Compare generated outputs against the immutable records under `results/pure_flair_p1_corrected_finsler/`.

Before using any result in the manuscript, verify the distributed code and
evidence snapshot:

```bash
python tools/verify_environment.py
python tools/release_manifest.py
python -m unittest discover -s tests -v
```

`RELEASE_MANIFEST.json` is the authoritative release-level integrity record.
Per-analysis provenance files are retained as historical execution records and
are not silently rewritten when scripts are subsequently cleaned or relocated.
Regenerate the release manifest with `python tools/release_manifest.py --write`
only after the numerical outputs have been deliberately revalidated.

Every program exposes its accepted paths and options through `--help` where applicable. The method uses anisotropic NewMetric diffusion; Gaussian filtering is not part of the reported procedure.

## Licences and archival release

Original software code is released under the MIT License; see `LICENSE`.
Original non-image documentation and derived numerical research outputs are
released under CC BY 4.0; see `LICENSE-DATA.md`. Source MR images and reference
masks are not redistributed. Third-party code and materials retain their
original terms, as detailed in the licence files.

The bundled `facseg/src/facseg` utilities are co-authored project code and are
included in this release under the MIT License with the authors' permission.
The separate proprietary FAC reference package is not included.

Versioned releases are preserved in Zenodo. Cite the version-specific DOI shown
on the corresponding GitHub release or Zenodo record.

## Citation

Release metadata are provided in `CITATION.cff`. The first public archival
software release is version 1.0.0. The repository citation metadata will be
updated with the Zenodo DOI after the first deposition; no placeholder DOI is
used.
