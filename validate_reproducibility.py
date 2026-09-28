from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
checks=[]
def check(name,condition,detail=''):
    checks.append({'check':name,'passed':bool(condition),'detail':str(detail)})

raw=pd.read_csv(ROOT/'data/raw/Dataset_Original.csv',sep=';',dtype=str,low_memory=False)
panel=pd.read_csv(ROOT/'data/processed/Dataset_PlayerSeason_Modelling.csv',low_memory=False)
idx=pd.read_csv(ROOT/'data/processed/Role_Specific_Index_Dataset.csv',low_memory=False)
feat=pd.read_csv(ROOT/'data/processed/Longitudinal_Features_Dataset.csv',low_memory=False)
oof=pd.read_csv(ROOT/'results/predictions/04_out_of_fold_predictions.csv')
pooled=pd.read_csv(ROOT/'results/tables/04_regression_metrics_pooled.csv')
tuning=pd.read_csv(ROOT/'results/tables/04_hyperparameter_tuning.csv')
ab=pd.read_csv(ROOT/'results/tables/05_feature_group_ablation_results.csv')
abtune=pd.read_csv(ROOT/'results/tables/05_ablation_hyperparameter_tuning.csv')
abpred=pd.read_csv(ROOT/'results/predictions/05_ablation_predictions.csv')
sel=pd.read_csv(ROOT/'results/tables/06_selection_coverage.csv')
direct=pd.read_csv(ROOT/'results/predictions/06_direct_classification_predictions.csv')
interval=pd.read_csv(ROOT/'results/predictions/06_prediction_interval_predictions.csv')
feature_dict=pd.read_csv(ROOT/'results/tables/03_feature_dictionary.csv')
sens900=pd.read_csv(ROOT/'results/tables/02_target_900min_sensitivity.csv')

check('Raw dataset shape',raw.shape==(22929,178),raw.shape)
check('Player-season panel row count',len(panel)==21371,len(panel))
check('No duplicate player-season keys',not panel.duplicated(['player_id','season']).any(),panel.duplicated(['player_id','season']).sum())
check('Role target row count',len(idx)==21371,len(idx))
check('450-minute eligible target count',idx['index_450min'].notna().sum()==14661,idx['index_450min'].notna().sum())
check('900-minute sensitivity target exists',idx['index_900min_sensitivity'].notna().sum()>0,idx['index_900min_sensitivity'].notna().sum())
check('900-minute sensitivity summary complete',set(sens900['group'])=={'Overall','DF','MF','FW'},sens900[['group','n']].to_dict('records'))
check('Longitudinal feature shape',feat.shape==(21371,59),feat.shape)
check('Canonical predictor dictionary has 45 unique features',len(feature_dict)==45 and feature_dict['feature'].nunique()==45,len(feature_dict))
check('First-season historical count is zero',feat.loc[feat['season_idx']==0,'n_hist_seasons'].fillna(-1).eq(0).all())
check('OOF row count',len(oof)==4742,len(oof))
check('OOF player-season keys unique',not oof.duplicated(['player_id','season']).any(),oof.duplicated(['player_id','season']).sum())
check('OOF fold sizes',oof.groupby('fold').size().tolist()==[1212,1200,1187,1143],oof.groupby('fold').size().to_dict())
check('Target season labels',oof.groupby('fold')['target_season'].first().tolist()==['2021/22','2022/23','2023/24','2024/25'],oof.groupby('fold')['target_season'].first().to_dict())
check('Forecast origin labels',oof.groupby('fold')['forecast_origin_season'].first().tolist()==['2020/21','2021/22','2022/23','2023/24'],oof.groupby('fold')['forecast_origin_season'].first().to_dict())

pred_cols=['pred_persistence','pred_role_age_league','pred_linear','pred_ridge','pred_elasticnet','pred_random_forest','pred_xgboost','pred_lightgbm','pred_mixed_blup']
check('All primary OOF predictions finite',np.isfinite(oof[pred_cols].to_numpy(float)).all())
check('Mixed-effects prediction covers every OOF row',len(oof)==oof['pred_mixed_blup'].notna().sum(),oof['pred_mixed_blup'].notna().sum())
check('Unseen players receive zero BLUP adjustment',np.allclose(oof.loc[~oof['seen_in_train'].astype(bool),'blup_adjustment'],0.0))

# One selected hyperparameter configuration per tunable model and outer fold.
selected=tuning[tuning['selected'].astype(bool)]
expected_models={'Ridge','ElasticNet','Random Forest','XGBoost','LightGBM'}
counts=selected.groupby(['model','fold']).size()
check('One selected primary tuning configuration per model × fold',set(selected['model'])==expected_models and len(counts)==20 and (counts==1).all(),counts.to_dict())

# Ablation is independently tuned: every model × fold × design × specification has one selected row.
absel=abtune[abtune['selected'].astype(bool)]
abcounts=absel.groupby(['model','fold','design','spec']).size()
expected_ab_groups=abtune[['model','fold','design','spec']].drop_duplicates().shape[0]
check('One selected ablation tuning configuration per model × fold × specification',len(abcounts)==expected_ab_groups and (abcounts==1).all(),f'{len(abcounts)} groups')

xgb=float(pooled.loc[pooled.model=='XGBoost','MAE'].iloc[0]); ridge=float(pooled.loc[pooled.model=='Ridge','MAE'].iloc[0])
abfull=ab[(ab.design=='leave_one_out') & (ab.spec=='full')]
check('Ablation full XGBoost matches primary model',abs(float(abfull.loc[abfull.model=='XGBoost','MAE'].iloc[0])-xgb)<1e-10)
check('Ablation full Ridge matches primary model',abs(float(abfull.loc[abfull.model=='Ridge','MAE'].iloc[0])-ridge)<1e-10)
# Full-ablation predictions must reproduce the primary predictions fold-by-fold.
# The ablation prediction table is intentionally compact and does not duplicate row identifiers,
# so equality is checked in the preserved outer-test row order within each fold.
for model,col in [('XGBoost','pred_xgboost'),('Ridge','pred_ridge')]:
    ok=True; details=[]
    for fold in sorted(oof['fold'].unique()):
        main=oof.loc[oof['fold']==fold,col].to_numpy(float)
        abl=abpred[(abpred['design']=='leave_one_out')&(abpred['spec']=='full')&(abpred['model']==model)&(abpred['fold']==fold)]['y_pred'].to_numpy(float)
        same=len(main)==len(abl) and np.allclose(main,abl)
        ok=ok and same; details.append((int(fold),len(main),len(abl),bool(same)))
    check(f'Full-ablation {model} predictions reproduce primary predictions',ok,details)
check('XGBoost is lowest pooled MAE',pooled.sort_values('MAE').iloc[0]['model']=='XGBoost',pooled.sort_values('MAE').head(3)[['model','MAE']].to_dict('records'))

selection=dict(zip(sel['selection_outcome'],sel['n']))
check('Selection candidate count',int(sel.n.sum())==7416,int(sel.n.sum()))
check('Selection included count',selection.get('Included in evaluation')==4742,selection)
check('Direct classification prediction count',len(direct)==4742,len(direct))
check('Prediction-interval prediction count',len(interval)==4742,len(interval))
interval_cols=[c for c in interval.columns if c.endswith('_lower') or c.endswith('_upper')]
check('Prediction interval bounds finite',np.isfinite(interval[interval_cols].to_numpy(float)).all(),interval_cols)
# lower <= upper for each method present
pairs=[]
for c in interval_cols:
    if c.endswith('_lower'):
        u=c.replace('_lower','_upper')
        if u in interval.columns: pairs.append((c,u))
check('Prediction interval lower bounds do not exceed upper bounds',all((interval[l]<=interval[u]).all() for l,u in pairs),pairs)

required_tables=[
'01_data_audit_summary.csv','01_identity_audit_summary.csv','01_season_retention_summary.csv',
'02_target_distribution_by_role.csv','02_target_league_means.csv','02_target_900min_sensitivity.csv',
'03_feature_dictionary.csv','03_feature_availability_summary.csv','04_regression_metrics_pooled.csv',
'04_mixed_effects_blup_summary.csv','05_leave_one_group_out_summary.csv','06_direct_classification_metrics.csv',
'06_prediction_interval_metrics.csv','06_season_specific_top20_results.csv','06_selection_coverage.csv']
check('Required report-ready tables exist',all((ROOT/'results/tables'/f).exists() for f in required_tables),[f for f in required_tables if not (ROOT/'results/tables'/f).exists()])
required_figures=['02_target_distribution_by_role.png','04_pooled_regression_mae.png','04_xgboost_mae_by_role.png','05_nested_feature_group_ablation.png','06_xgboost_residual_distribution.png','06_direct_xgb_confusion_matrix.png','06_top20_mean_overlap.png','06_prediction_interval_coverage.png']
check('Required report-ready figures exist',all((ROOT/'results/figures'/f).exists() for f in required_figures),[f for f in required_figures if not (ROOT/'results/figures'/f).exists()])

for nbpath in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(nbpath,4)
    errs=[]; exec_counts=[]
    for cell in nb.cells:
        if cell.cell_type=='code':
            if cell.execution_count is not None: exec_counts.append(cell.execution_count)
            errs.extend([o for o in cell.get('outputs',[]) if o.output_type=='error'])
    check(f'Notebook executed without stored errors: {nbpath.name}',len(errs)==0 and len(exec_counts)>0,f'errors={len(errs)}, executed_cells={len(exec_counts)}')

out=pd.DataFrame(checks)
failed=out[~out.passed]
print(out.to_string(index=False))
print(f'\nPassed {out.passed.sum()} / {len(out)} reproducibility checks.')
if len(failed):
    raise SystemExit(f'Reproducibility checks failed: {failed.check.tolist()}')
