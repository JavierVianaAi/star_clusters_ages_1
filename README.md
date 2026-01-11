# Star Cluster Age Experiments

This repository contains the full experimental pipeline used to study star cluster age prediction from PHANGS-HST imaging, with a particular focus on masking strategies, normalization choices, and control experiments. The code is organized to reproduce baseline metrics, single-case experiments, large parameter sweeps, and all figures used in the associated paper.

---

## 1. Data requirements (IMPORTANT)

All notebooks and scripts in this repository require access to the raw PHANGS dataset.

You must provide a directory called raw_data containing the following file:

raw_data/
raw_phangs_dataset.h5

The file raw_phangs_dataset.h5 contains all star cluster data used in this work.

If this file is missing or the path is incorrect, the notebooks and scripts will not run.

Make sure the path to raw_data is correctly specified in your environment or within the scripts before execution.

---

## 2. Repository structure and purpose

### 0_baseline_case

Baseline reference experiments.

This folder is used to compute baseline performance metrics on the train, validation, and test sets using simple non-learned predictors.

Specifically, it evaluates:
- Median age guess
- Random guessing

These results provide lower-bound reference metrics against which all learned models are compared.

Main files:
- get_baseline_metrics.py
- get_baseline_metrics.ipynb
- get_baseline_metrics.sh

---

### 0_preliminary_studies

Exploratory normalization experiments.

This folder contains notebooks used to study the effect of different normalization strategies on the input images. These preliminary studies informed the normalization choices adopted in the main experiments.

Files:
- example_normalizations_1im_case.ipynb
- example_normalizations_5im_case.ipynb

---

### 1_single_case

Single masking experiments.

This folder is used to run one specific masking configuration at a time.

For running a single masking case using the stacked white-light image:
- single_case_1im.py

For running a single masking case using the five-filter images:
- single_case_5im.py

These scripts are useful for debugging, qualitative inspection, and testing new masking configurations before launching large parameter sweeps.

This directory also includes SLURM scripts, output logs, and legacy exploratory code stored in 0_old.

---

### wrong_targets_to_check

Sanity-check experiments.

This folder contains code demonstrating that the models do not learn meaningful structure when the target ages are randomly shuffled. In this case, the models converge to random guessing around the mean, as expected.

These experiments serve as a control, confirming that the learning signal observed in the main experiments is physically meaningful and not an artifact of the training procedure.

---

### 2_many_case

Large parameter sweep experiments.

This folder contains scripts used to run ranges of masking configurations in order to systematically evaluate model performance as a function of mask size.

For range runs using the stacked white-light image:
- many_case_1im.py

For range runs using the five-filter images:
- many_case_5im.py

These experiments produce the core quantitative results used in the paper.

The folder includes batch execution scripts, SLURM outputs, and stored results for downstream analysis.

---

### 3_plots

Figure generation for the paper.

This folder contains notebooks used to generate all figures and plots appearing in the paper, including bar plots, comparison plots, and range visualizations.

Some subfolders store intermediate versions of plotting code to ensure full reproducibility of the figures.

This directory is used purely for analysis and visualization and does not run model training.

---

### 4_example_models

This folder contains example outputs from trained models for selected experimental configurations. It is intended for inspection, visualization, and qualitative comparison of model behavior, rather than for running new training jobs.

Each subfolder corresponds to a specific experimental setup. The folder names explicitly encode the configuration used, including:

- Image type (1im for stacked white-light image, 5im for five-filter images)
- Whether extreme ages are removed (remextremes)
- Whether data augmentation is used (yes_augment or no_augment)
- Whether blackout masking is applied
- The normalization strategy (inner, outer, no_normby)
- The image modality (single-image or five-images)

Inside each configuration folder, results are organized by mask radius (for example, R_6), and include:

- Heatmaps for individual instances
- Histograms of predictions for individual instances, both normalized and unnormalized
- Histograms of model outputs
- Scatter plots summarizing final metrics across all trained models
- Serialized metrics files (pkl) for downstream analysis

Each configuration typically contains multiple trained models (model_0, model_1, etc.), each with:

- Training, validation, and test prediction versus true-value plots
- Loss evolution plots
- Stored prediction arrays and final metrics

This folder is provided as a compact reference showcasing representative model behavior and outputs used in the analysis and figures of the paper.

---

## 3. Typical workflow

A typical workflow using this repository is:

1. Ensure the raw data file is available:
raw_data/raw_phangs_dataset.h5

2. Compute baseline metrics:
0_baseline_case

3. Optional: inspect normalization effects:
0_preliminary_studies

4. Run single masking experiments:
single_case_1im.py
single_case_5im.py

5. Run full masking sweeps:
many_case_1im.py
many_case_5im.py

6. Generate all paper figures:
3_plots

---

## 4. Notes

- SLURM scripts are provided for cluster execution.
- Output logs are stored in the corresponding slurm_outs directories.
- The repository prioritizes reproducibility and traceability; some redundancy in code and outputs is intentional.
