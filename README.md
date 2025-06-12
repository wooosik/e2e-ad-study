# e2e-ad-study

E2E Autonomous Driving Study

This repository contains code snippets and configuration files for the
E2E Autonomous Driving study plan. Start by creating the conda
environment and installing pre-commit hooks for code formatting.

```bash
conda env create -f environment.yml
conda activate e2e-ad-study
pre-commit install
```

Verify that PyTorch sees your GPU by running:

```bash
python check_gpu.py
```

The `study_plan.py` script contains minimal examples for later weeks of
the study. Set the environment variable `SAMPLE_IMAGE` to an RGB image
path to test the frame-to-tensor conversion example.
