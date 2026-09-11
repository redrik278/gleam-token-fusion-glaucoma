from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix

O = Path("output/gleam_fourclass")
F = Path("submission_medical_image_analysis/figures")
F.mkdir(parents=True, exist_ok=True)
models = ["early_fusion", "late_fusion", "gated_token", "official_hamm"]
labels = {"early_fusion":"Early fusion", "late_fusion":"Late fusion", "gated_token":"Proposed", "official_hamm":"HAMM reproduction"}
colors = {"early_fusion":"#8d99ae", "late_fusion":"#6c757d", "gated_token":"#0068b5", "official_hamm":"#d98e04"}
classes = ["Normal", "Early", "Intermediate", "Advanced"]

plt.rcParams.update({"font.size":9, "axes.titlesize":10, "axes.labelsize":9})

# Training dynamics from validation histories. Epoch counts differ because of early stopping.
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
fields = [("loss","Training loss"), ("auc_macro_ovr","Validation macro AUC"),
          ("accuracy","Validation accuracy"), ("ece10","Validation ECE")]
for ax, (field, title) in zip(axes.flat, fields):
    for seed, ls in zip((17,29,43), ("-","--",":")):
        h = json.loads((O/f"gated_token_seed{seed}_history.json").read_text())
        ax.plot([r["epoch"] for r in h], [r[field] for r in h], ls, marker="o", ms=3,
                color="#0068b5", label=f"Seed {seed}")
        best = json.loads((O/f"gated_token_seed{seed}_result.json").read_text())["best_epoch"]
        ax.axvline(best, color="#9ca3af", lw=.6, alpha=.35)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
axes[0,0].legend(ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F/"training_dynamics.png", dpi=300); plt.close(fig)

def ensemble(model):
    ds=[pd.read_csv(O/f"{model}_seed{s}_test.csv") for s in (17,29,43)]
    p=np.mean([d[[f"p{i}" for i in range(4)]].to_numpy() for d in ds],axis=0)
    return ds[0]["label"].to_numpy(),p

# Per-class ROC curves for all directly compared ensembles.
fig, axes = plt.subplots(2,2,figsize=(10.5,7.2))
for k, ax in enumerate(axes.flat):
    for m in models:
        y,p=ensemble(m); fpr,tpr,_=roc_curve((y==k).astype(int),p[:,k]); a=auc(fpr,tpr)
        ax.plot(fpr,tpr,lw=1.8,color=colors[m],label=f"{labels[m]} ({a:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_title(classes[k]); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
axes[1,1].legend(frameon=False,fontsize=7,loc="lower right")
fig.tight_layout(); fig.savefig(F/"per_class_roc.png",dpi=300); plt.close(fig)

# Per-stage recall and F1 reveal class imbalance hidden by accuracy.
rec=np.zeros((len(models),4)); f1=np.zeros_like(rec)
for i,m in enumerate(models):
    y,p=ensemble(m); pred=p.argmax(1)
    _,rec[i],f1[i],_=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),sharey=True); x=np.arange(4); w=.19
for i,m in enumerate(models):
    axes[0].bar(x+(i-1.5)*w,rec[i],w,label=labels[m],color=colors[m]); axes[1].bar(x+(i-1.5)*w,f1[i],w,label=labels[m],color=colors[m])
for ax,title in zip(axes,["Recall by reference stage","F1 score by reference stage"]):
    ax.set_xticks(x,classes,rotation=15); ax.set_ylim(0,1); ax.set_title(title); ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Score"); axes[1].legend(frameon=False,fontsize=7,ncol=2,loc="lower right")
fig.tight_layout(); fig.savefig(F/"per_stage_performance.png",dpi=300); plt.close(fig)

# Prediction-level confidence, entropy, and ordinal error for the proposed ensemble.
y,p=ensemble("gated_token"); pred=p.argmax(1); conf=p.max(1); entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(4); err=np.abs(pred-y)
fig,axes=plt.subplots(1,3,figsize=(11,3.8))
axes[0].hist(conf[pred==y],bins=np.linspace(.25,1,16),alpha=.75,label="Correct",color="#2a9d8f"); axes[0].hist(conf[pred!=y],bins=np.linspace(.25,1,16),alpha=.70,label="Incorrect",color="#e76f51"); axes[0].set_title("Maximum confidence"); axes[0].set_xlabel("Predicted probability"); axes[0].set_ylabel("Samples"); axes[0].legend(frameon=False)
axes[1].boxplot([entropy[y==k] for k in range(4)],tick_labels=classes,showfliers=False); axes[1].set_title("Normalized predictive entropy"); axes[1].tick_params(axis="x",rotation=18); axes[1].set_ylim(0,1)
counts=np.bincount(err,minlength=4); axes[2].bar(range(4),counts,color=["#2a9d8f","#e9c46a","#f4a261","#e76f51"]); axes[2].set_xticks(range(4)); axes[2].set_xlabel("Absolute stage error"); axes[2].set_ylabel("Samples"); axes[2].set_title("Ordinal error magnitude"); axes[2].bar_label(axes[2].containers[0])
for ax in axes: ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15)
fig.tight_layout(); fig.savefig(F/"prediction_level_analysis.png",dpi=300); plt.close(fig)

# Multi-metric component ablation from the saved seed-29 runs.
abl=pd.read_csv(O/"component_ablations_exploratory.csv")
full=json.loads((O/"gated_token_seed29_result.json").read_text())["test"]
rows=[("Complete",full)]
for _,r in abl.iterrows():
    name={"token_no_gates":"No gates","token_no_transformer":"No Transformer","token_no_regularizers":"No regularizers"}[r["variant"]]
    rows.append((name,r.to_dict()))
metrics=[("auc_macro_ovr","Macro AUC"),("accuracy","Accuracy"),("f1_macro","Macro F1"),("quadratic_kappa","QWK")]
fig,axes=plt.subplots(2,2,figsize=(9.5,6.2)); cs=["#0068b5","#8ecae6","#8ecae6","#8ecae6"]
for ax,(key,title) in zip(axes.flat,metrics):
    vals=[r[key] for _,r in rows]; bars=ax.bar([n for n,_ in rows],vals,color=cs); ax.set_ylim(max(0,min(vals)-.06),min(1,max(vals)+.025)); ax.set_title(title); ax.tick_params(axis="x",rotation=15); ax.bar_label(bars,fmt="%.3f",fontsize=8); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15)
fig.suptitle("Seed-29 component ablations",y=.995); fig.tight_layout(); fig.savefig(F/"ablation_multimetric.png",dpi=300); plt.close(fig)

print("created training_dynamics.png, per_class_roc.png, per_stage_performance.png, prediction_level_analysis.png, ablation_multimetric.png")
