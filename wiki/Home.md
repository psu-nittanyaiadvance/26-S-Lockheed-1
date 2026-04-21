# Wiki — 26-S-Lockheed-1

Physics-informed 3D Gaussian Splatting for underwater imaging sonar.

## Pages

| Page | Description |
|------|-------------|
| [Overview](Overview.md) | Research motivation, what makes sonar different from RGB, two-model summary |
| [Physics](Physics.md) | All loss functions and derivations: beam pattern, Gamma NLL, elevation constraint, reflectivity regulariser |
| [Architecture](Architecture.md) | Code structure, key classes and functions, gradient paths, patched files |
| [Datasets](Datasets.md) | All datasets with paths, frame counts, sonar hardware parameters |
| [Training](Training.md) | Full training commands for every dataset and model, hyperparameter reference |
| [Results](Results.md) | All benchmark runs, ablations, key findings |
| [Implementation Notes](Implementation.md) | Bugs fixed, diagnostic print reference, pitfalls to avoid |

---

## Quick reference

```bash
# SonarSplat v2 — any dataset, any steps
conda activate sonarsplat
cd sonar_splat
bash scripts/run_v2.sh <dataset_name> <results_dir> <steps>

# Z-Splat v2 — AONeuS
cd z_splatting
bash scripts/run_aoneus_v2.sh <results_dir> <steps>
```

Add `--z_loss_weight 0.0` to either command for an L1 baseline ablation.
