import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score, f1_score
R='ml/results/v2/'
m=pd.read_csv('ml/results/test_idx_frame_mapping.csv')
y=m.label.values; files=m.file.values
def load(mod): return np.array([pd.read_csv(f'{R}{mod}_s{s}_test_preds.csv').p_anomaly.values for s in (1,2,3)])
P1,P2=load('cnn1d'),load('cnn2d')
b=pd.read_csv(R+'baseline_energy_test_preds.csv').p_anomaly.values
thr=json.load(open(R+'baseline_energy_metrics.json'))['threshold']
def met(p,thr_=0.5):
    pr=(p>=thr_).astype(int); return dict(acc=(pr==y).mean(),f1=f1_score(y,pr),auc=roc_auc_score(y,p),fp=int(((pr==1)&(y==0)).sum()),fn=int(((pr==0)&(y==1)).sum()))
rows={}
for name,P in (('CNN1D',P1),('CNN2D',P2)):
    per=[met(p) for p in P]
    rows[name+' (3 seed ort. ± std)']={k:(np.mean([d[k] for d in per]),np.std([d[k] for d in per],ddof=1)) for k in ('acc','f1','auc','fp','fn')}
    rows[name+' (3 seed ensemble)']={k:(v,np.nan) for k,v in met(P.mean(0)).items()}
rows['Enerji esigi (baseline)']={k:(v,np.nan) for k,v in met(b,thr).items()}
# cluster bootstrap by recording (test = 60 recordings) for ensemble & baseline
rng=np.random.default_rng(0); recs=np.unique(files); idx_by={r:np.where(files==r)[0] for r in recs}
def boot(p,thr_,n=2000):
    out={'acc':[],'auc':[],'f1':[]}
    for _ in range(n):
        s=rng.choice(recs,len(recs)); ii=np.concatenate([idx_by[r] for r in s])
        if len(set(y[ii]))<2: continue
        pr=(p[ii]>=thr_).astype(int)
        out['acc'].append((pr==y[ii]).mean()); out['auc'].append(roc_auc_score(y[ii],p[ii])); out['f1'].append(f1_score(y[ii],pr))
    return {k:(np.percentile(v,2.5),np.percentile(v,97.5)) for k,v in out.items()}
ci={'CNN1D ensemble':boot(P1.mean(0),.5),'CNN2D ensemble':boot(P2.mean(0),.5),'Enerji esigi':boot(b,thr)}
lines=['# Nihai sonuçlar (donmuş split, test = 240 pencere / 60 kayıt)\n','Eşik 0.5 (baseline: eğitimden gelen eşik %.3f). Üç seed: 1, 2, 3.\n'%thr,
'| Model | Doğruluk | F1 | ROC-AUC | FP | FN |','|---|---|---|---|---|---|']
for k,d in rows.items():
    f=lambda key,dec=3:('%.*f ± %.*f'%(dec,d[key][0],dec,d[key][1]) if not np.isnan(d[key][1]) else '%.*f'%(dec,d[key][0]))
    lines.append(f'| {k} | {f("acc")} | {f("f1")} | {f("auc")} | {f("fp",1)} | {f("fn",1)} |')
lines+=['','## %95 güven aralığı (kayıt bazlı bootstrap, 2000 tekrar, 60 test kaydı)\n','| Model | Doğruluk | F1 | ROC-AUC |','|---|---|---|---|']
for k,d in ci.items(): lines.append(f'| {k} | [{d["acc"][0]:.3f}, {d["acc"][1]:.3f}] | [{d["f1"][0]:.3f}, {d["f1"][1]:.3f}] | [{d["auc"][0]:.3f}, {d["auc"][1]:.3f}] |')
# breakdown by SNR/SIR for 2D & 1D ensembles
pr1=(P1.mean(0)>=.5).astype(int); pr2=(P2.mean(0)>=.5).astype(int)
m['e1']=(pr1!=y); m['e2']=(pr2!=y); m['eb']=((b>=thr).astype(int)!=y)
lines+=['','## Hata kırılımı (ensemble, eşik 0.5): OnlyLTE ve SNR\n','| SNR (dB) | OnlyLTE pencere | CNN1D FP | CNN2D FP | Baseline FP |','|---|---|---|---|---|']
o=m[m.label==0]
for s,d in o.groupby('snr_db'): lines.append(f'| {s:g} | {len(d)} | {d.e1.sum()} | {d.e2.sum()} | {d.eb.sum()} |')
lines+=['','## Hata kırılımı: LTE+DSSS ve SIR\n','| SIR (dB) | LTE+DSSS pencere | CNN1D FN | CNN2D FN | Baseline FN |','|---|---|---|---|---|']
c=m[m.label==1]
for s,d in c.groupby('sir_db'): lines.append(f'| {s:g} | {len(d)} | {d.e1.sum()} | {d.e2.sum()} | {d.eb.sum()} |')
lines+=['','## Model boyutu ve CPU gecikmesi (Colab CPU, batch=1, model only)\n','| Model | Parametre | CPU ms/örnek |','|---|---|---|','| CNN1D | 85,730 | 145.5 ± 17.2 |','| CNN2D | 60,706 | 92.4 ± 0.3 |','| Enerji eşiği | 0 | 2.0 |']
open('ml/results/final_results.md','w',encoding='utf-8').write('\n'.join(lines)); print('\n'.join(lines))
