# Repository structure

```text
csmpr-player-performance-prediction/
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── run_all.py
├── validate_reproducibility.py
├── build_manifest.py
├── analysis_manifest.json
├── MANIFEST_SHA256.csv
├── config/
│   └── model_config.json
├── src/
│   ├── __init__.py
│   └── analysis_utils.py
├── data/
│   ├── README.md
│   ├── raw/                 # source CSV downloaded locally; ignored by Git
│   └── processed/           # generated datasets; ignored by Git
├── notebooks/
│   ├── 01_data_audit_and_player_season_panel.ipynb
│   ├── 02_role_specific_performance_target.ipynb
│   ├── 03_longitudinal_feature_engineering.ipynb
│   ├── 04_chronological_model_evaluation.ipynb
│   ├── 05_feature_group_ablation.ipynb
│   └── 06_error_direction_practical_uncertainty.ipynb
└── results/
    ├── README.md
    ├── tables/              # aggregate/report-ready outputs tracked in Git
    ├── figures/             # generated report figures tracked in Git
    ├── predictions/         # row-level local outputs; ignored by Git
    └── logs/                # local execution logs; ignored by Git
```
