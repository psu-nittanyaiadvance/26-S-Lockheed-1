# Branch Merger Agent Memory

## Repository Structure
- **Main branch**: `main` -- contains only an empty README.md (commit 97c8f59)
- All feature branches share the same root commit: `97c8f59` ("first commit")
- `flSea_analysis` and `image_class` share a deeper common ancestor at `14c3fb1` ("other datasets")

## Branch Relationships
- `image_class` is a strict superset of `flSea_analysis` for shared files (download_and_rank.py, download_dataset.py, rank_images.py, requirements.txt are identical)
- `sonar-splat` is fully independent -- no code overlap with other branches
- When merging flSea_analysis + image_class, always prefer image_class versions of differing files (more comprehensive)

## Key Files and Purposes
- `download_and_rank.py`: Harvard Dataverse download + CLIP/CV incremental top-K ranking (1306 lines)
- `rank_images.py`: Standalone two-pass image ranker (1081 lines) -- shares CV heuristic code with download_and_rank.py
- `preprocess_datasets.py`: CLIP zero-shot classification (14 categories) + artifact detection + cleaning (1475 lines)
- `vehicle_artifact_crop.py`: Generic artifact detection via median compositing (1076 lines)
- `sunboat_crop.py` / `sunboat_templatematching.py`: Specialized Sunboat mount detection (~600 lines each)
- `sonar_splat/`: SonarSplat framework (~1791 files, CUDA extensions, based on gsplat)

## Code Duplication Patterns
- CV heuristics (entropy, laplacian_var, saturation_penalty, edge_density) are duplicated between rank_images.py and download_and_rank.py
- `load_image_safe()` appears in rank_images.py, download_and_rank.py, and preprocess_datasets.py with minor variations
- CLIP model loading pattern is similar across all three but serves different purposes (ranking vs classification)

## Dependencies
- Data pipeline: torch, transformers, ultralytics, opencv, PIL, scipy, numpy, pandas, matplotlib, py7zr
- SonarSplat: torch+CUDA, gsplat (custom CUDA build), open3d, wandb, nerfview, viser, torchmetrics, fused-ssim

## Merge Strategy Used (March 2026)
1. Created `unified-pipeline` from `main`
2. Fast-forward merged `image_class` (superset of flSea_analysis)
3. Merged `sonar-splat` (independent, only .gitignore conflict)
4. Cherry-picked unique files from `flSea_analysis` (sonar_analysis_report.txt, sonar_dataset_analysis.png)
5. Consolidated .gitignore from all branches
6. Wrote comprehensive README.md

## Dataset Configs (hardcoded paths in preprocess_datasets.py)
- Paths are absolute and machine-specific (e.g., /Users/ethanknox/Desktop/...)
- Three datasets configured: flsea_vi, shipwreck, sunboat
