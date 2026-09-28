# Data

## Source

**Top 5 League Football Player Stats (2017–2025)** — Kaggle dataset by Emre Yılmaz:  
https://www.kaggle.com/datasets/emrey3lmaz/top-5-league-football-player-stats-2017-2025

The analysis expects the original semicolon-delimited CSV at:

`data/raw/Dataset_Original.csv`

The source dataset is not distributed through the Git repository. Download it from the Kaggle data card and place it at the path above before running the pipeline. Any redistribution should follow the licence/usage conditions stated by the dataset publisher on Kaggle.

## Generated data

Notebooks 01–03 regenerate these local files in `data/processed/`:

- `Dataset_PlayerSeason_Modelling.csv`
- `Role_Specific_Index_Dataset.csv`
- `Longitudinal_Features_Dataset.csv`

These generated datasets are ignored by Git because they are reproducible outputs of the analysis.
