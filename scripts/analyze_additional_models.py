"""Aggregate ordinal and modality-specific augmentation experiments."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score,f1_score,roc_auc_score,average_precision_score

OUT=Path('output/gleam_fourclass');SEEDS=(17,29,43)
MODELS=('gated_token','gated_token_clinical_aug','ordinal_token')
def ensemble(model):
 frames=[pd.read_csv(OUT/f'{model}_seed{s}_test.csv',dtype={'id':str}) for s in SEEDS];base=frames[0]
 for f in frames[1:]:
  assert np.array_equal(base.id.to_numpy(),f.id.to_numpy()) and np.array_equal(base.label.to_numpy(),f.label.to_numpy())
 return base.label.to_numpy(),np.mean([f[[f'p{i}' for i in range(4)]].to_numpy() for f in frames],0)
def metrics(y,p):
 pred=p.argmax(1)
 return {'macro_auc':roc_auc_score(y,p,multi_class='ovr',average='macro'),'macro_auprc':np.mean([average_precision_score(y==c,p[:,c]) for c in range(4)]),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'qwk':cohen_kappa_score(y,pred,weights='quadratic')}
ens={};y=None
for m in MODELS:
 yy,ens[m]=ensemble(m);y=yy if y is None else y;assert np.array_equal(y,yy)
point={m:metrics(y,p) for m,p in ens.items()};rng=np.random.default_rng(20260913);B=10000;boots={m:[] for m in MODELS};deltas={m:[] for m in MODELS[1:]}
for _ in range(B):
 ix=rng.integers(0,len(y),len(y));yy=y[ix]
 if len(np.unique(yy))<4:continue
 values={m:roc_auc_score(yy,p[ix],multi_class='ovr',average='macro') for m,p in ens.items()}
 for m,v in values.items():boots[m].append(v)
 for m in MODELS[1:]:deltas[m].append(values[m]-values['gated_token'])
rows=[];paired={}
basepred=ens['gated_token'].argmax(1)
for m in MODELS:
 ci=np.quantile(boots[m],[.025,.975]);rows.append({'model':m,**point[m],'macro_auc_ci95_low':ci[0],'macro_auc_ci95_high':ci[1]})
 if m!='gated_token':
  pred=ens[m].argmax(1);b=int(((pred==y)&(basepred!=y)).sum());c=int(((pred!=y)&(basepred==y)).sum());paired[m]={'auc_delta_model_minus_original':point[m]['macro_auc']-point['gated_token']['macro_auc'],'auc_delta_ci95':np.quantile(deltas[m],[.025,.975]).tolist(),'model_only_correct':b,'original_only_correct':c,'mcnemar_exact_p':float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.0}
pd.DataFrame(rows).to_csv(OUT/'additional_model_summary.csv',index=False);(OUT/'additional_model_statistics.json').write_text(json.dumps({'seeds':SEEDS,'bootstrap_replicates':B,'paired':paired},indent=2));print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(paired,indent=2))
