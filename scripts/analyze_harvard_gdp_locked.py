"""Analyze the locked Harvard-GDP patient-independent experiment."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,f1_score,brier_score_loss,log_loss,confusion_matrix
from scipy.stats import binomtest
O=Path('output/harvard_gdp_locked');META=Path('data/public/Harvard-GDP/data_summary.csv');SEEDS=(17,29,43,71,101);MODELS=('oct','vf','late','reliability')
def ens(k,split):
 d=[pd.read_csv(O/f'{k}_seed{s}_{split}.csv').sort_values('id') for s in SEEDS];return d[0].id.to_numpy(),d[0].label.to_numpy(),np.mean([x.probability.to_numpy() for x in d],0)
def met(y,p,t):
 q=p>=t;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return {'auc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'f1':float(f1_score(y,q)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'ppv':float(tp/(tp+fp)) if tp+fp else None,'npv':float(tn/(tn+fn)) if tn+fn else None,'brier':float(brier_score_loss(y,p)),'nll':float(log_loss(y,np.c_[1-p,p]))}
def threshold(y,p):
 vals=np.unique(p);z=[(met(y,p,t)['sensitivity']+met(y,p,t)['specificity']-1,t) for t in vals];return float(max(z)[1])
def paired(y,a,b,B=10000,seed=20260916):
 rng=np.random.default_rng(seed);base=roc_auc_score(y,a)-roc_auc_score(y,b);v=[]
 for _ in range(B):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))==2:v.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
 qa=a>=.5;qb=b>=.5;ca=qa==y;cb=qb==y;x=int(np.sum(ca&~cb));z=int(np.sum(~ca&cb));return {'auc_delta':float(base),'ci95':np.quantile(v,[.025,.975]).tolist(),'mcnemar_p':float(binomtest(min(x,z),x+z,.5).pvalue) if x+z else 1.,'correct_a_only':x,'correct_b_only':z}
def main():
 out={'protocol':'publication/LOCKED_HARVARD_GDP_PROTOCOL_2026-09-16.md','models':{},'comparisons':{},'subgroups':{}};T={}
 for k in MODELS:
  _,yv,pv=ens(k,'val');ids,y,p=ens(k,'test');t=threshold(yv,pv);out['models'][k]={'validation_threshold':t,'test':met(y,p,t)};T[k]=(ids,y,p,t)
 y=T['reliability'][1]
 for k in ('oct','vf','late'):out['comparisons']['reliability_minus_'+k]=paired(y,T['reliability'][2],T[k][2])
 meta=pd.read_csv(META).set_index('filename');ids=T['reliability'][0];m=meta.loc[ids]
 for col in ('gender','race'):
  out['subgroups'][col]={}
  for g,ix in m.groupby(col).groups.items():
   pos=m.index.get_indexer(ix);yy=y[pos];pp=T['reliability'][2][pos]
   if len(np.unique(yy))==2:out['subgroups'][col][str(g)]={'n':len(pos),**met(yy,pp,T['reliability'][3])}
 (O/'locked_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
