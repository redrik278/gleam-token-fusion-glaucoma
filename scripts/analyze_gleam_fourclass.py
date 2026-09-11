"""Publication-oriented analysis for repeated four-class GLEAM predictions."""
from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score
from scipy.stats import binomtest

O=Path("output/gleam_fourclass");O.mkdir(parents=True,exist_ok=True);MAIN=["early_fusion","late_fusion","gated_token","official_hamm"];SEEDS=[17,29,43]
all_results=[json.loads(p.read_text()) for p in O.glob("*_result.json")];pd.json_normalize(all_results).to_csv(O/"all_run_metrics.csv",index=False)
ens={};y=None;ids=None
for m in MAIN:
 ps=[]
 for s in SEEDS:
  d=pd.read_csv(O/f"{m}_seed{s}_test.csv",dtype={"id":str});ids=d.id.str.zfill(4);y=d.label.to_numpy();ps.append(d[[f"p{i}" for i in range(4)]].to_numpy())
 ens[m]=np.mean(ps,0)
pd.DataFrame([{"model":m,"macro_auc":roc_auc_score(y,p,multi_class="ovr",average="macro"),"accuracy":accuracy_score(y,p.argmax(1))} for m,p in ens.items()]).to_csv(O/"ensemble_performance.csv",index=False)

rng=np.random.default_rng(20260910);B=5000;boot={m:[] for m in MAIN};delta={m:[] for m in MAIN if m!="gated_token"}
for _ in range(B):
 ix=rng.integers(0,len(y),len(y));yy=y[ix]
 if len(np.unique(yy))<4:continue
 a={m:roc_auc_score(yy,p[ix],multi_class="ovr",average="macro") for m,p in ens.items()}
 for m in MAIN:boot[m].append(a[m])
 for m in delta:delta[m].append(a["gated_token"]-a[m])
stats={"bootstrap_replicates":B,"macro_auc":{},"paired_delta_proposed_minus_comparator":{},"mcnemar_accuracy":{}}
for m in MAIN:stats["macro_auc"][m]={"estimate":float(roc_auc_score(y,ens[m],multi_class="ovr",average="macro")),"ci95":[float(x) for x in np.quantile(boot[m],[.025,.975])]}
gp=ens["gated_token"].argmax(1)
for m in delta:
 q=np.asarray(delta[m]);other=ens[m].argmax(1);b=int(((gp==y)&(other!=y)).sum());c=int(((gp!=y)&(other==y)).sum())
 stats["paired_delta_proposed_minus_comparator"][m]={"estimate":float(stats["macro_auc"]["gated_token"]["estimate"]-stats["macro_auc"][m]["estimate"]),"ci95":[float(x) for x in np.quantile(q,[.025,.975])]}
 stats["mcnemar_accuracy"][m]={"proposed_only_correct":b,"comparator_only_correct":c,"exact_p":float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.0}
(O/"statistical_comparisons.json").write_text(json.dumps(stats,indent=2))

# Calibration and confusion matrices for seed ensembles.
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for m,p in ens.items():
 conf=p.max(1);ok=(p.argmax(1)==y);xs=[];ys=[]
 for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
  q=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9))
  if q.any():xs.append(conf[q].mean());ys.append(ok[q].mean())
 ax[0].plot(xs,ys,'o-',label=m)
ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].set(xlabel="Confidence",ylabel="Observed accuracy",title="Multiclass calibration");ax[0].legend(fontsize=8,frameon=False)
ths=np.linspace(.01,.50,100);yb=(y>0).astype(int)
for m,p in ens.items():
 pb=1-p[:,0];nb=[]
 for t in ths:
  pr=pb>=t;tp=((pr)&(yb==1)).sum();fp=((pr)&(yb==0)).sum();nb.append(tp/len(y)-fp/len(y)*t/(1-t))
 ax[1].plot(ths,nb,label=m)
ax[1].plot(ths,[yb.mean()-((yb==0).mean())*t/(1-t) for t in ths],'--',label='treat all',color='gray');ax[1].axhline(0,color='black',lw=.8);ax[1].set(xlabel="Threshold probability",ylabel="Net benefit",title="Any-glaucoma decision curve");ax[1].legend(fontsize=8,frameon=False)
fig.tight_layout();fig.savefig(O/"calibration_decision_curve.png",dpi=300);plt.close(fig)
fig,axs=plt.subplots(1,4,figsize=(14,3.3))
for ax,(m,p) in zip(axs,ens.items()):
 cm=confusion_matrix(y,p.argmax(1),labels=range(4));im=ax.imshow(cm,cmap='Blues');ax.set_title(m,fontsize=9);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_xticks(range(4));ax.set_yticks(range(4))
 for i in range(4):
  for j in range(4):ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=8)
fig.tight_layout();fig.savefig(O/"confusion_matrices.png",dpi=300);plt.close(fig)

# Computational cost and exploratory component ablations.
rows=[]
for m in MAIN:
 rr=[json.loads((O/f"{m}_seed{s}_result.json").read_text()) for s in SEEDS];rows.append({"model":m,"parameters":rr[0]["parameters"],"training_seconds_mean":np.mean([r["training_seconds"] for r in rr]),"peak_gpu_mb_mean":np.mean([r["peak_gpu_mb"] for r in rr]),"note":""})
pd.DataFrame(rows).to_csv(O/"computational_cost.csv",index=False)
abl=[r for r in all_results if r["model"].startswith("token_")];pd.DataFrame([{"variant":r["model"],"seed":r["seed"],**r["test"]} for r in abl]).to_csv(O/"component_ablations_exploratory.csv",index=False)

# Local cohort: cross-endpoint sensitivity only; no specificity/AUC with one normal.
sf=pd.read_csv("data/processed_v2/structural_functional_labels.csv",dtype={"patient_id":str});oof=pd.read_csv("output/cross_validation_v2/oof_predictions.csv",dtype={"patient_id":str});oof=oof[oof.architecture=="full"].merge(sf[["patient_id","primary_label_status"]],on="patient_id",how="inner");pos=oof[oof.primary_label_status=="glaucoma"]
local={"definite_positive_oof_n":int(len(pos)),"sensitivity_at_0.5":float((pos.probability>=.5).mean()) if len(pos) else None,"scope":"exploratory cross-endpoint sensitivity of the historical local OOF model; specificity and AUC not estimable"};(O/"local_external_sensitivity.json").write_text(json.dumps(local,indent=2))
print(json.dumps(stats,indent=2));print(json.dumps(local,indent=2))
