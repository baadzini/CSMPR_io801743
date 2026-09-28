from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import json
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

SEASON_ORDER = ['1718','1819','1920','2021','2122','2223','2324','2425']
SEASON_LABELS = ['2017/18','2018/19','2019/20','2020/21','2021/22','2022/23','2023/24','2024/25']
SEASON_IDX = {s: i for i, s in enumerate(SEASON_ORDER)}

OUTER_FOLDS = [
    {'fold': 1, 'train': [0,1,2],       'test': 3},
    {'fold': 2, 'train': [0,1,2,3],     'test': 4},
    {'fold': 3, 'train': [0,1,2,3,4],   'test': 5},
    {'fold': 4, 'train': [0,1,2,3,4,5], 'test': 6},
]

FEATURE_GROUPS = {
    'current_performance': [
        'perf_rank_t','perf_z_t',
    ],
    'trajectory_history': [
        'perf_z_t_minus1','trend_1s','trend_2s','trend_3s','n_hist_seasons',
    ],
    'age_exposure': [
        'age_','age_sq','minutes_t','starts_t','minutes_t_minus1','minutes_volatility_3s',
    ],
    'role': [
        'is_DF','is_MF','is_FW',
    ],
    'context_availability': [
        'league_strength_t','club_strength_t','club_strength_change',
        'team_attack_strength_t','team_defense_strength_t',
        'missed_matches_t','missed_matches_t_minus1',
        'transfer_indicator','league_change_indicator',
    ],
    'role_conditional_output': [
        'goals_per90','xG_per90','assists_per90','xA_per90',
        'goals_per90_x_FW','goals_per90_x_MF','goals_per90_x_DF',
        'xG_per90_x_FW','xG_per90_x_MF','xG_per90_x_DF',
        'assists_per90_x_FW','assists_per90_x_MF','assists_per90_x_DF',
        'xA_per90_x_FW','xA_per90_x_MF','xA_per90_x_DF',
        'progression_per90','creation_per90','possession_per90','defense_per90',
    ],
}

FEATURE_COLS = sum(FEATURE_GROUPS.values(), [])
TARGET = 'rank_t1'


def repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path.cwd()).resolve()
    if p.name == 'notebooks':
        return p.parent
    if (p / 'notebooks').exists() and (p / 'data').exists():
        return p
    for parent in [p, *p.parents]:
        if (parent / 'notebooks').exists() and (parent / 'data').exists():
            return parent
    raise RuntimeError('Could not locate repository root.')


def paths(root: Optional[Path] = None) -> Dict[str, Path]:
    root = repo_root(root)
    return {
        'root': root,
        'raw': root / 'data' / 'raw',
        'processed': root / 'data' / 'processed',
        'tables': root / 'results' / 'tables',
        'figures': root / 'results' / 'figures',
        'predictions': root / 'results' / 'predictions',
        'logs': root / 'results' / 'logs',
        'config': root / 'config',
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def lead_series(df: pd.DataFrame, col: str, k: int = 1) -> np.ndarray:
    """Exact k-season-ahead lookup for the same player; gaps remain missing."""
    piv = df.pivot_table(index='player_id', columns='season_idx', values=col, aggfunc='first')
    lead_df = df[['player_id','season_idx']].copy()
    lead_df['lookup_idx'] = lead_df['season_idx'] + k
    try:
        stacked = piv.stack(future_stack=True).rename(col).reset_index().rename(columns={'season_idx':'lookup_idx'})
    except TypeError:
        stacked = piv.stack(dropna=False).rename(col).reset_index().rename(columns={'season_idx':'lookup_idx'})
    merged = lead_df.merge(stacked, on=['player_id','lookup_idx'], how='left')
    return merged[col].to_numpy()


def build_supervised_sample(feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add t+1 target/role and return (full annotated table, same-role eligible transitions)."""
    d = feat.copy()
    d['rank_t1'] = lead_series(d, 'perf_rank_t', 1)
    d['role_t1'] = lead_series(d, 'role', 1)
    elig = d[d['perf_rank_t'].notna() & d['rank_t1'].notna() & (d['role'] == d['role_t1'])].copy()
    return d, elig


def split_outer(elig: pd.DataFrame, fold: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = elig[elig['season_idx'].isin(fold['train'])].copy()
    test = elig[elig['season_idx'] == fold['test']].copy()
    return train, test


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    columns: List[str]
    medians: Dict[str, float]
    missing_flag_sources: List[str]
    dropped_all_missing: List[str]
    dropped_constant: List[str]
    scaler: Optional[StandardScaler]
    X_train: np.ndarray
    X_test: np.ndarray


def prepare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Iterable[str],
    *,
    scale: bool = False,
    add_missing_flags: bool = True,
) -> PreparedData:
    """Training-only preprocessing: drop all-missing/constant columns, learn medians, optionally scale.

    Missing-indicator selection is based only on the training partition. Test values never determine
    which columns or indicators enter the model.
    """
    tr = train.copy()
    te = test.copy()
    cols = list(cols)

    dropped_all_missing = [c for c in cols if c not in tr.columns or tr[c].notna().sum() == 0]
    base_cols = [c for c in cols if c not in dropped_all_missing]

    medians: Dict[str, float] = {}
    flag_sources: List[str] = []
    for c in base_cols:
        med = float(tr[c].median())
        medians[c] = med
        if add_missing_flags and tr[c].isna().any():
            flag = f'{c}__missing'
            tr[flag] = tr[c].isna().astype(np.int8)
            te[flag] = te[c].isna().astype(np.int8)
            flag_sources.append(c)
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)

    candidate_cols = base_cols + [f'{c}__missing' for c in flag_sources]
    dropped_constant = [c for c in candidate_cols if tr[c].nunique(dropna=False) <= 1]
    final_cols = [c for c in candidate_cols if c not in dropped_constant]

    Xtr = tr[final_cols].to_numpy(dtype=float)
    Xte = te[final_cols].to_numpy(dtype=float)
    scaler = None
    if scale:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

    return PreparedData(
        train_df=tr, test_df=te, columns=final_cols, medians=medians,
        missing_flag_sources=flag_sources, dropped_all_missing=dropped_all_missing,
        dropped_constant=dropped_constant, scaler=scaler, X_train=Xtr, X_test=Xte,
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    ae = np.abs(yt - yp)
    residual = yt - yp
    if len(yt) == 0:
        return {k: np.nan for k in ['n','MAE','MedianAE','RMSE','R2','Spearman','Bias','P90_AE','P95_AE']}
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - np.sum((yt - yp) ** 2) / ss_tot if ss_tot > 0 else np.nan
    rho = stats.spearmanr(yt, yp).statistic if len(yt) > 2 else np.nan
    return {
        'n': int(len(yt)),
        'MAE': float(ae.mean()),
        'MedianAE': float(np.median(ae)),
        'RMSE': float(np.sqrt(np.mean((yt - yp) ** 2))),
        'R2': float(r2),
        'Spearman': float(rho),
        'Bias': float(residual.mean()),
        'P90_AE': float(np.quantile(ae, 0.90)),
        'P95_AE': float(np.quantile(ae, 0.95)),
    }


def model_metric_row(y_true, y_pred, model: str, fold: Optional[int] = None, **extra) -> Dict:
    out = {'model': model}
    if fold is not None:
        out['fold'] = fold
    out.update(extra)
    out.update(regression_metrics(y_true, y_pred))
    return out


def age_bucket_baseline(age: pd.Series) -> pd.Series:
    bins = [0,21,24,27,30,33,99]
    labels = ['<21','21-23','24-26','27-29','30-32','33+']
    return pd.cut(age, bins=bins, labels=labels, right=False)


def predict_role_age_league_mean(train: pd.DataFrame, test: pd.DataFrame, target: str = TARGET) -> np.ndarray:
    tr = train.copy(); te = test.copy()
    tr['age_bucket'] = age_bucket_baseline(tr['age_'])
    te['age_bucket'] = age_bucket_baseline(te['age_'])
    grp_mean = tr.groupby(['role','age_bucket','league_primary'], observed=True)[target].mean()
    role_age_mean = tr.groupby(['role','age_bucket'], observed=True)[target].mean()
    role_mean = tr.groupby('role', observed=True)[target].mean()
    global_mean = float(tr[target].mean())
    preds = []
    for _, row in te.iterrows():
        key = (row['role'], row['age_bucket'], row['league_primary'])
        if key in grp_mean.index and pd.notna(grp_mean[key]):
            preds.append(float(grp_mean[key])); continue
        key2 = (row['role'], row['age_bucket'])
        if key2 in role_age_mean.index and pd.notna(role_age_mean[key2]):
            preds.append(float(role_age_mean[key2])); continue
        if row['role'] in role_mean.index:
            preds.append(float(role_mean[row['role']])); continue
        preds.append(global_mean)
    return np.asarray(preds)


def inner_chronological_split(outer_train: pd.DataFrame, fold: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use the final season available in the outer-training window as inner validation."""
    val_idx = max(fold['train'])
    inner_train = outer_train[outer_train['season_idx'] < val_idx].copy()
    inner_val = outer_train[outer_train['season_idx'] == val_idx].copy()
    return inner_train, inner_val


def concat_unique(groups: Iterable[str]) -> List[str]:
    out: List[str] = []
    for g in groups:
        for c in FEATURE_GROUPS[g]:
            if c not in out:
                out.append(c)
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)
