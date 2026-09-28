from pathlib import Path
import hashlib, json, platform, sys
from datetime import datetime, timezone
import pandas as pd

ROOT=Path(__file__).resolve().parent
ANALYSIS_VERSION='1.0.0'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def excluded_common(rel):
    return ('__pycache__/' in rel or rel.startswith('results/logs/') or 'checkpoint' in rel.lower() or rel.startswith('.git/'))

def repository_included(rel):
    if excluded_common(rel): return False
    if rel in {'MANIFEST_SHA256.csv','ARCHIVE_MANIFEST_SHA256.csv','analysis_manifest.json'}: return False
    if rel.startswith('data/raw/') and not rel.endswith('.gitkeep'): return False
    if rel.startswith('data/processed/') and not rel.endswith('.gitkeep'): return False
    if rel.startswith('results/predictions/') and not rel.endswith('.gitkeep'): return False
    return True

def archive_included(rel):
    if excluded_common(rel): return False
    if rel in {'MANIFEST_SHA256.csv','ARCHIVE_MANIFEST_SHA256.csv','analysis_manifest.json'}: return False
    return True

repo_files=[]; archive_files=[]
for path in sorted(ROOT.rglob('*')):
    if not path.is_file(): continue
    rel=path.relative_to(ROOT).as_posix()
    row={'path':rel,'bytes':path.stat().st_size,'sha256':sha256(path)}
    if repository_included(rel): repo_files.append(row)
    if archive_included(rel): archive_files.append(row)

pd.DataFrame(repo_files).to_csv(ROOT/'MANIFEST_SHA256.csv',index=False)
pd.DataFrame(archive_files).to_csv(ROOT/'ARCHIVE_MANIFEST_SHA256.csv',index=False)
summary_file=ROOT/'results/tables/06_final_analysis_summary.json'
summary=json.loads(summary_file.read_text()) if summary_file.exists() else {}
manifest={
    'analysis_version':ANALYSIS_VERSION,
    'generated_utc':datetime.now(timezone.utc).isoformat(),
    'python_version':sys.version,
    'platform':platform.platform(),
    'data_source':{
        'title':'Top 5 League Football Player Stats (2017-2025)',
        'publisher':'Emre Yılmaz (Kaggle)',
        'url':'https://www.kaggle.com/datasets/emrey3lmaz/top-5-league-football-player-stats-2017-2025',
        'local_input':'data/raw/Dataset_Original.csv',
        'raw_input_sha256':sha256(ROOT/'data/raw/Dataset_Original.csv'),
        'included_in_git':False,
    },
    'notebook_order':[p.name for p in sorted((ROOT/'notebooks').glob('*.ipynb'))],
    'final_analysis_summary':summary,
    'repository_manifest_entries':len(repo_files),
    'archive_manifest_entries':len(archive_files),
}
(ROOT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
