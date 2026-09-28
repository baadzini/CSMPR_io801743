from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    '01_data_audit_and_player_season_panel.ipynb',
    '02_role_specific_performance_target.ipynb',
    '03_longitudinal_feature_engineering.ipynb',
    '04_chronological_model_evaluation.ipynb',
    '05_feature_group_ablation.ipynb',
    '06_error_direction_practical_uncertainty.ipynb',
]

raw = ROOT / 'data' / 'raw' / 'Dataset_Original.csv'
if not raw.exists():
    raise FileNotFoundError(
        f'Missing canonical raw input: {raw}. Place Dataset_Original.csv there before running the pipeline.'
    )

for name in NOTEBOOKS:
    path = ROOT / 'notebooks' / name
    print(f'\n=== Executing {name} ===', flush=True)
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        str(path), '--inplace', '--ExecutePreprocessor.timeout=3600'
    ], cwd=ROOT, check=True)

print('\n=== Running reproducibility checks ===', flush=True)
subprocess.run([sys.executable, str(ROOT / 'validate_reproducibility.py')], cwd=ROOT, check=True)

print('\n=== Building analysis manifest ===', flush=True)
subprocess.run([sys.executable, str(ROOT / 'build_manifest.py')], cwd=ROOT, check=True)
print('\nPipeline complete.', flush=True)
