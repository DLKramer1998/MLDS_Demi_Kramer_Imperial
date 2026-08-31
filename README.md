# A Bayesian Framework for Structured Label Corruption

Code and executed notebooks accompanying the MSc Machine Learning and Data Science dissertation:

**A Bayesian Framework for Structured Label Corruption: Recoverability and Localisation under Covariate-Dependent Misclassification**

**Author:** Demi Kramer  
**Supervisor:** Nick Heard  
**Imperial College London, Department of Mathematics**  
**August 2026**

## Overview

This repository contains the final computational analysis for a Bayesian latent-variable framework for structured, covariate-dependent label corruption. The model jointly estimates a latent response relationship, a halfspace corruption mechanism, and observation-level posterior probabilities of label corruption.

The empirical work uses the one-sided `1 -> 0` corruption mechanism described in the dissertation. The simulation study separates predictive robustness, observation ranking, individual posterior uncertainty, and full mechanism recovery, and treats posterior diagnostics as part of the inferential conclusion.

## Repository contents

- `chapter0_framework_validation.ipynb` — controlled framework validation, soft-boundary sensitivity, and supplied/general-boundary checks.
- `chapter0b_prior_sensitivity.ipynb` — targeted paired prior-sensitivity analysis.
- `chapter1_simulation.ipynb` — replicated simulation experiments. Experiments 1–3 use 20 independently generated datasets per setting; learned-boundary experiments use 10 datasets per condition.
- `chapter1b_mode_exploration.ipynb` — Experiment 4d forced-exploration follow-up: three representative failing learned-boundary fits refitted with eight chains initialised at both posterior modes.
- `chapter2_applied.ipynb` — BRFSS diabetes and UCI credit-default semi-synthetic applications, including pre-corruption response-model checks and diagnostic-gated Bayesian analyses.
- `corruption_model_axis_FINAL.stan` — axis-supplied corruption-boundary model.
- `corruption_model_general_FINAL.stan` — learned general-boundary model.
- `corruption_model_axis_FIXED_C_DIAGNOSTIC.stan` — fixed-threshold diagnostic model used in the applied analysis.
- `data/` — datasets used by the applied notebook.

The notebooks are included with their executed outputs. Random seeds are fixed in the analysis code.

## Data

The applied analyses use:

1. **2015 BRFSS diabetes health indicators** — derived from the US Centers for Disease Control and Prevention Behavioral Risk Factor Surveillance System.
2. **Default of Credit Card Clients** — Yeh and Lien (2009), UCI Machine Learning Repository.

The repository includes the exact CSV files used by the final applied notebook so that its relative paths resolve without modification.

## Environment

The analysis was developed with Python 3.11 and uses Stan through `cmdstanpy`.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

CmdStan must also be installed. If it is not already available:

```python
from cmdstanpy import install_cmdstan
install_cmdstan()
```

The Chapter 1 notebook also contains a guarded installation block and can run either locally or in Google Colab.

## Running the analysis

Run notebooks from the repository root so the relative data paths resolve correctly. A natural order is:

1. `chapter0_framework_validation.ipynb`
2. `chapter0b_prior_sensitivity.ipynb`
3. `chapter1_simulation.ipynb`
4. `chapter1b_mode_exploration.ipynb`
5. `chapter2_applied.ipynb`

Generated high-replication results are written beneath `high_replication/`. That directory is excluded from version control because the notebooks already contain the executed outputs and regenerate these files when run.

## Diagnostic policy

Bayesian parameter-recovery summaries are interpreted only for fits satisfying the prespecified diagnostic gate used in the dissertation: \(\hat R \leq 1.05\), bulk effective sample size at least 100, no divergent transitions, and no maximum-tree-depth events. Diagnostic failures are retained as scientific outcomes rather than silently discarded.

## Reproducibility note

The standalone Stan files are provided for transparency. Some notebooks also write the relevant Stan programs into their own safe output directories before compilation so that notebook execution remains self-contained.
