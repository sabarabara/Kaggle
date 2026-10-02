"""Build an executable, portable notebook; preserve the user's original EDA."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# S6E10 — 特徴量エンジニアリングのための分析

既存の `aaaaaaaaaanya.ipynb` を補完する分析。全件で分布を調べ、同じ分割で特徴量群を比較します。
元の列を残したうえで候補を追加します。満足率の図は関連を示し、因果効果を示しません。

「評価0＝未利用/該当なし」は未確認の仮説です。件数の少ないセルは保留し、
全件EDAから考えた特徴量のCVは探索用として扱います。最終採用前には別のseedや全件で再検証します。

公式: [データ](https://www.kaggle.com/competitions/playground-series-s6e10/data) /
[評価方法](https://www.kaggle.com/competitions/playground-series-s6e10/overview/evaluation)''')
code('''from pathlib import Path
import sys, os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ローカル/Kaggle両方でデータを探す。Kaggleでは本ファイルを追加して使うか、
# 次のセルの特徴量関数（埋め込み）だけで実行可能。
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p/'data/raw/train.csv').exists()), Path.cwd())
DATA = next((p for p in [ROOT/'data/raw',
    Path('/kaggle/input/competitions/playground-series-s6e10'),
    Path('/kaggle/input/playground-series-s6e10')] if (p/'train.csv').exists()), None)
if DATA is None: raise FileNotFoundError('train.csv/test.csv を配置してください')
OUT = ROOT/'reports'; OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
SEED = 42
pd.set_option('display.max_columns', 30)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .2})
train = pd.read_csv(DATA/'train.csv'); test = pd.read_csv(DATA/'test.csv')
raw_target = train['satisfaction']
if set(raw_target.dropna().unique()) <= {True, False, 0, 1}:
    train['satisfaction'] = raw_target.astype('int8')
elif set(raw_target.dropna().unique()) <= {'satisfied', 'neutral or dissatisfied'}:
    train['satisfaction'] = raw_target.eq('satisfied').astype('int8')
else: raise ValueError(f'Unexpected target labels: {raw_target.unique()}')
assert train['satisfaction'].nunique() == 2 and not train['satisfaction'].isna().any()
print('Train:', train.shape, 'Test:', test.shape)
print('Satisfaction rate:', train.satisfaction.mean())
''')
code((ROOT/'analysis/features.py').read_text())
code('''def save_table(df, name):
    df.to_csv(OUT/f'{name}.csv', index=False)
    return df

def rate_table(df, keys):
    t = df.groupby(keys, dropna=False, observed=True)['satisfaction'].agg(n='size', rate='mean').reset_index()
    # Wilson 95% intervals, including values exactly 0 or 1.
    z = 1.96; n = t['n']; p = t['rate']; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    t['low'], t['high'] = center-half, center+half
    return t

def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUT/'figures'/f'{name}.png', bbox_inches='tight')
    plt.show(); plt.close(fig)
''')
md('''## 1. 品質・分布差・IDの診断
重複はIDを除いて確認。テストとの重複はラベルを参照せず特徴量だけを比較します。
分布差は全列共通の Jensen–Shannon divergence（0〜1、大きいほど差）で比較。
連続値の区切りは訓練データの分位点を使います。KSのp値は標本が大きいため採否の基準にはしません。''')
code('''from scipy.spatial.distance import jensenshannon
features = train.columns.drop(['id','satisfaction']).tolist()
quality = pd.DataFrame({'feature': features,
    'dtype': [str(train[c].dtype) for c in features],
    'train_missing': [train[c].isna().sum() for c in features],
    'test_missing': [test[c].isna().sum() for c in features],
    'train_unique': [train[c].nunique() for c in features]})
display(save_table(quality, 'quality'))
htrain = pd.util.hash_pandas_object(train[features], index=False)
htest = pd.util.hash_pandas_object(test[features], index=False)
print('Duplicate training feature rows:', htrain.duplicated().sum())
print('Training rows in conflicting-label duplicate groups:',
    train.groupby(htrain)['satisfaction'].transform('nunique').gt(1).sum())
print('Test rows matching training features (hash check):', htest.isin(set(htrain)).sum())
shift_rows = []
for c in features:
    a, b = train[c], test[c]
    ks = np.nan
    if pd.api.types.is_numeric_dtype(a) and a.nunique() > 20:
        edges = np.unique(a.quantile(np.linspace(0,1,21)))
        edges[0], edges[-1] = -np.inf, np.inf
        aa = pd.cut(a, edges).astype(str).where(a.notna(), '__NA__')
        bb = pd.cut(b, edges).astype(str).where(b.notna(), '__NA__')
        ks = ks_2samp(a.dropna(), b.dropna()).statistic
    else:
        aa, bb = a.fillna('__NA__').astype(str), b.fillna('__NA__').astype(str)
    levels = sorted(set(aa.unique()) | set(bb.unique()))
    pa = aa.value_counts(normalize=True).reindex(levels, fill_value=0)
    pb = bb.value_counts(normalize=True).reindex(levels, fill_value=0)
    shift_rows.append({'feature':c, 'js_divergence':jensenshannon(pa,pb,base=2)**2, 'ks':ks})
shift = pd.DataFrame(shift_rows).sort_values('js_divergence', ascending=False)
display(save_table(shift, 'distribution_shift'))
fig, ax = plt.subplots(figsize=(10,5))
ax.barh(shift.feature.iloc[:12][::-1], shift.js_divergence.iloc[:12][::-1])
ax.set_xlabel('Train/test Jensen-Shannon divergence'); finish(fig, 'distribution_shift')
idview = train[['id','satisfaction']].copy()
idview['id_decile'] = pd.qcut(idview['id'], 10)
display(save_table(rate_table(idview, ['id_decile']), 'id_deciles'))
''')
md('''## 2. サービス評価は単調か？ 特に0の挙動
評価ごとの満足率と95%区間を比較。相関係数だけでは0の特殊な挙動や閾値を取りこぼします。
0を除いた平均や0フラグを候補にしますが、低評価と未利用のどちらかはデータだけでは確定できません。''')
code('''rating_tables = []
fig, axes = plt.subplots(5,3,figsize=(15,17))
for c, ax in zip(RATINGS, axes.ravel()):
    t = rate_table(train, [c]); t['feature'] = c
    t = t.rename(columns={c:'value'}); rating_tables.append(t)
    ax.errorbar(t.value, t.rate, yerr=[t.rate-t.low,t.high-t.rate], fmt='o-')
    ax.set_title(c,fontsize=10); ax.set_ylim(0,1); ax.set_xticks(range(6))
    ax.set_xlabel('Rating'); ax.set_ylabel('Satisfaction rate')
for ax in axes.ravel()[len(RATINGS):]: ax.axis('off')
finish(fig, 'rating_response')
ratings_long = pd.concat(rating_tables, ignore_index=True)
save_table(ratings_long, 'rating_response')
display(ratings_long.query('value == 0')[['feature','n','rate','low','high']])
zero_view = train.assign(zero_count=train[RATINGS].eq(0).sum(axis=1))
display(save_table(rate_table(zero_view, ['zero_count']), 'zero_count_response'))
''')
md('''## 3. 顧客層と評価の相互作用
同じ評価値でも旅行目的、クラス、顧客タイプで満足率が違うか確認。
少数セルは伏せ、件数表も保存します。差は他の特徴量による交絡を含むため、CVで追加効果を確認します。''')
code('''segment_tables = []
for c in CATEGORIES:
    t = rate_table(train,[c]); t['feature'] = c
    t = t.rename(columns={c:'value'}); segment_tables.append(t)
display(save_table(pd.concat(segment_tables,ignore_index=True), 'category_response'))
display(save_table(rate_table(train,['Customer Type','Type of Travel','Class']), 'customer_segments'))
interaction_tables = []
fig, axes = plt.subplots(2,3,figsize=(15,9))
for row, segment in enumerate(['Type of Travel','Class']):
    for col, rating in enumerate(['Inflight wifi service','Online boarding','Inflight entertainment']):
        t = rate_table(train,[segment,rating]); t['segment_feature']=segment; t['rating_feature']=rating
        interaction_tables.append(t.rename(columns={segment:'segment',rating:'rating'}))
        pivot = t.assign(rate=t.rate.where(t.n >= 200)).pivot(index=segment,columns=rating,values='rate')
        ax=axes[row,col]; im=ax.imshow(pivot, vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(pivot.columns)),pivot.columns)
        ax.set_yticks(range(len(pivot.index)),pivot.index,fontsize=8)
        ax.set_title(f'{rating} by {segment}',fontsize=10)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                if pd.notna(pivot.iloc[i,j]): ax.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if pivot.iloc[i,j]<.5 else 'black')
        fig.colorbar(im,ax=ax)
finish(fig,'segment_interactions')
save_table(pd.concat(interaction_tables,ignore_index=True), 'segment_interactions')
''')
md('''## 4. 評価の一致・食い違いと遅延
合成データでは似た項目の一致自体に情報がある可能性があります。
一致フラグ、評価差、評価ペアのカテゴリを別の表現として候補化。
遅延は差分・回復量・ゼロ・欠損フラグを調べます。元の遅延列は残します。''')
code('''pairs = []
fig, axes = plt.subplots(2,3,figsize=(15,10))
for i, ((a,b),ax) in enumerate(zip(PAIRS,axes.ravel())):
    t=rate_table(train,[a,b]); t['pair']=f'{a} / {b}'
    pairs.append(t.rename(columns={a:'left',b:'right'}))
    pivot=t.assign(rate=t.rate.where(t.n>=200)).pivot(index=a,columns=b,values='rate')
    im=ax.imshow(pivot,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns); ax.set_yticks(range(len(pivot.index)),pivot.index)
    ax.set_xlabel(b,fontsize=9); ax.set_ylabel(a,fontsize=9); fig.colorbar(im,ax=ax)
finish(fig,'rating_pairs')
save_table(pd.concat(pairs,ignore_index=True),'rating_pairs')
equality_rows=[]
for a,b in PAIRS:
    view=train.assign(equal=train[a].eq(train[b]))
    t=rate_table(view,['equal']); t['pair']=f'{a} / {b}'
    t['test_equal_rate']=test[a].eq(test[b]).mean()
    equality_rows.append(t)
display(save_table(pd.concat(equality_rows,ignore_index=True),'pair_equality'))
delays=train[['satisfaction','Departure Delay in Minutes','Arrival Delay in Minutes']].copy()
delays['delay_delta']=delays['Arrival Delay in Minutes']-delays['Departure Delay in Minutes']
delays['delay_delta_bin']=pd.cut(delays.delay_delta,[-np.inf,-15,-1,0,1,15,np.inf])
display(save_table(rate_table(delays,['delay_delta_bin']),'delay_delta_response'))
delays['arrival_missing']=delays['Arrival Delay in Minutes'].isna()
display(save_table(rate_table(delays,['arrival_missing']),'missing_response'))
for c in ['Age','Flight Distance','Departure Delay in Minutes']:
    v=train.assign(bin=pd.qcut(train[c],10,duplicates='drop'))
    print(c); display(rate_table(v,['bin']))
''')
md('''## 5. 特徴量群ごとの交差検証（一次スクリーニング）

速度のため訓練から層化抽出した120,000行で3-fold。全候補で同じ行・分割・seed・モデルを使用。
CatBoost 400木・深さ6を固定し、検証スコアで反復数を選ばない設定です。
ID・目的変数を特徴量から除外し、候補はすべて行単位で作成。ターゲットエンコーディングは使用しません。

候補の選定に全件EDAを使っているため、これは探索用比較であり未見データでの最終評価ではありません。
小さな差は未確定です。採用には全件・追加seed・別モデルでの再現が必要。
PCAやt-SNEの見た目の分離だけを理由に特徴量を採用しません。''')
code('''CV_N = min(120_000,len(train))
if CV_N < len(train):
    sample_idx,_=train_test_split(np.arange(len(train)), train_size=CV_N,
        stratify=train.satisfaction, random_state=SEED)
else: sample_idx=np.arange(len(train))
sample=train.iloc[sample_idx].reset_index(drop=True)
y=sample.satisfaction.to_numpy()
folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(sample,y))
experiments={'raw':[], **{k:[k] for k in feature_groups(sample)},
             'all':list(feature_groups(sample))}
results=[]; oofs={}; importances=[]
for name,families in experiments.items():
    X,cats=model_frame(sample,families)
    oof=np.zeros(len(sample)); scores=[]
    for fold,(itr,iva) in enumerate(folds):
        start=time.time()
        model=CatBoostClassifier(iterations=400,depth=6,learning_rate=.08,
            loss_function='Logloss',random_seed=SEED+fold,thread_count=6,
            verbose=False,allow_writing_files=False)
        model.fit(X.iloc[itr],y[itr],cat_features=cats)
        p=model.predict_proba(X.iloc[iva])[:,1]; oof[iva]=p
        auc=roc_auc_score(y[iva],p); scores.append(auc)
        results.append({'experiment':name,'fold':fold,'auc':auc,'seconds':time.time()-start,'features':X.shape[1]})
        if name in ['raw','all']:
            importances.append(pd.DataFrame({'experiment':name,'fold':fold,
                'feature':X.columns,'importance':model.feature_importances_}))
        print(f'{name:13s} fold={fold} AUC={auc:.6f} seconds={time.time()-start:.1f}',flush=True)
        pd.DataFrame(results).to_csv(OUT/'cv_folds.csv',index=False)
    oofs[name]=oof
    np.savez_compressed(OUT/f'oof_{name}.npz',id=sample.id.to_numpy(),y=y,pred=oof)
fold_results=pd.DataFrame(results)
summary=fold_results.groupby('experiment').agg(auc_mean=('auc','mean'),auc_std=('auc','std'),features=('features','first'))
summary['oof_auc']=[roc_auc_score(y,oofs[name]) for name in summary.index]
summary['delta_oof']=summary.oof_auc-summary.loc['raw','oof_auc']
paired=fold_results.pivot(index='fold',columns='experiment',values='auc')
for name in summary.index:
    delta=paired[name]-paired['raw']
    summary.loc[name,'delta_fold_min']=delta.min(); summary.loc[name,'delta_fold_max']=delta.max()
summary=summary.sort_values('oof_auc',ascending=False)
display(save_table(summary.reset_index(),'cv_summary'))
importance=pd.concat(importances).groupby(['experiment','feature']).importance.mean().reset_index()
save_table(importance,'model_importance')
display(importance.query("experiment == 'all'").sort_values('importance',ascending=False).head(25))
fig,ax=plt.subplots(figsize=(9,4)); order=summary.index.tolist()
ax.barh(order[::-1],summary.loc[order[::-1],'delta_oof']); ax.axvline(0,color='black',lw=.8)
ax.set_xlabel('OOF AUC difference vs raw (exploratory)'); finish(fig,'cv_ablation')
''')
md('''## 6. 読み方と次の実験

- `rating_response.csv`：0や閾値の挙動。ただし満足率が高くても他の列を入れた後の追加効果は未確定。
- `segment_interactions.csv`：旅行目的・クラスで評価の意味が変わるか。件数と区間も確認。
- `rating_pairs.csv` / `pair_equality.csv`：一致や差の候補。強い関連があっても元の列でモデルが表現できる場合がある。
- `cv_summary.csv`：各群を単独追加した比較。複数foldで方向が一致する候補を優先。
- `model_importance.csv`：モデル内の使用量の目安。相関した列の重要度は分散するため、採否は追加比較で決める。

改善した候補は全件で深いモデル・追加seedへ進め、OOF予測を保存してモデル間の組み合わせを検証。
悪化した特徴量はこの設定での不採用候補であり、全モデルで無効と断定しません。''')
nb=nbf.v4.new_notebook(cells=cells)
nb.metadata={'kernelspec':{'display_name':'Kaggle analysis (Python 3.12)','language':'python','name':'kaggle-analysis'},
    'language_info':{'name':'python','version':'3.12'}}
dest=ROOT/'notebooks/feature_engineering/s6e10_feature_analysis.ipynb'
nbf.write(nb,dest)
print(dest)
