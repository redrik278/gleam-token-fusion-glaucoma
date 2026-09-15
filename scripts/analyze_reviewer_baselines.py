from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,f1_score,cohen_kappa_score
from scipy.stats import binomtest
O=Path('output/gleam_fourclass'); seeds=[17,29,43]
models=['vf','vf_oct','vf_oct_slo','weighted_logits','attention_pool','mlp_matched','token_no_aux','token_no_moddrop','gated_token','token_no_gates','late_fusion']
def ensemble(k):
 ds=[pd.read_csv(O/f'{k}_seed{s}_test.csv').sort_values('id') for s in seeds]
 ids=ds[0].id.astype(str).to_numpy();y=ds[0].label.to_numpy();p=np.mean([d[[f'p{i}' for i in range(4)]].to_numpy() for d in ds],0);return ids,y,p
def met(y,p):
 q=p.argmax(1);return dict(auc=float(roc_auc_score(y,p,multi_class='ovr',average='macro')),accuracy=float(accuracy_score(y,q)),balanced_accuracy=float(balanced_accuracy_score(y,q)),macro_f1=float(f1_score(y,q,average='macro')),qwk=float(cohen_kappa_score(y,q,weights='quadratic')))
D={k:ensemble(k) for k in models}; rows=[]
for k,(ids,y,p) in D.items():
 m=met(y,p); prm=json.loads((O/f'{k}_seed17_result.json').read_text())['parameters'];rows.append({'model':k,'parameters':prm,**m})
rng=np.random.default_rng(20260915)
def cmp(a,b,B=5000):
 _,y,pa=D[a];_,yb,pb=D[b];assert np.array_equal(y,yb);n=len(y);base=roc_auc_score(y,pa,multi_class='ovr',average='macro')-roc_auc_score(y,pb,multi_class='ovr',average='macro');vals=[]
 for _ in range(B):
  ix=rng.integers(0,n,n)
  if len(np.unique(y[ix]))<4:continue
  vals.append(roc_auc_score(y[ix],pa[ix],multi_class='ovr',average='macro')-roc_auc_score(y[ix],pb[ix],multi_class='ovr',average='macro'))
 qa=pa.argmax(1)==y;qb=pb.argmax(1)==y;b01=int(np.sum(qa&~qb));b10=int(np.sum(~qa&qb));mc=float(binomtest(min(b01,b10),b01+b10,.5).pvalue) if b01+b10 else 1.0
 return {'model_a':a,'model_b':b,'auc_delta_a_minus_b':float(base),'auc_delta_ci95':[float(x) for x in np.quantile(vals,[.025,.975])],'mcnemar_correct_a_only':b01,'mcnemar_correct_b_only':b10,'mcnemar_p_correctness':mc}
comparisons=[cmp('vf_oct','vf'),cmp('vf_oct_slo','vf_oct'),cmp('gated_token','weighted_logits'),cmp('gated_token','attention_pool'),cmp('gated_token','mlp_matched'),cmp('gated_token','token_no_gates'),cmp('gated_token','token_no_aux'),cmp('gated_token','token_no_moddrop')]
pd.DataFrame(rows).to_csv(O/'reviewer_baseline_ensemble_summary.csv',index=False);(O/'reviewer_baseline_comparisons.json').write_text(json.dumps(comparisons,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(comparisons,indent=2))
