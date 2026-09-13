"""Clinical operating metrics, missing-modality tests, and shortcut/XAI analyses."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.utils.data import DataLoader

spec=importlib.util.spec_from_file_location("gleam_train","scripts/train_gleam_fourclass.py")
gt=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gt;spec.loader.exec_module(gt)
OUT=Path("output/gleam_fourclass"); DATA=Path("data/public/GLEAM"); SEEDS=(17,29,43); MODS=("SLO","OCT","VF")
DEVICE=torch.device("cuda")

def load_model(seed):
    model=gt.Token().to(DEVICE); state=torch.load(OUT/f"gated_token_seed{seed}.pt",map_location=DEVICE,weights_only=True); model.load_state_dict(state["model"]); model.eval(); return model

def forward_masked(model,x,available=(0,1,2)):
    raw=[enc(x[:,i]) for i,enc in enumerate(model.e)]
    tokens=torch.stack([proj(z)*gate(z) for proj,gate,z in zip(model.p,model.g,raw)],1)+model.mod
    keep=torch.zeros(3,dtype=torch.bool,device=x.device);keep[list(available)]=True
    tokens=tokens.masked_fill((~keep)[None,:,None],0)
    fused=model.f(torch.cat([model.cls.expand(x.size(0),-1,-1),tokens],1))[:,0]
    return model.h(fused)

@torch.inference_mode()
def predict(models,loader,available=(0,1,2),perturb=None):
    ys=[];ids=[];allp=[[] for _ in models]
    for x,y,sid in loader:
        if perturb is not None:x=apply_perturbation(x,*perturb)
        x=x.to(DEVICE)
        for j,model in enumerate(models):
            with torch.autocast("cuda"):p=forward_masked(model,x,available).softmax(1)
            allp[j].append(p.float().cpu().numpy())
        ys.extend(y.numpy());ids.extend(sid)
    return np.asarray(ys),np.mean([np.concatenate(p) for p in allp],axis=0),ids

def roi_mask(modality,h=224,w=224):
    yy,xx=np.mgrid[:h,:w]
    if modality==0:return ((xx-w/2)**2+(yy-h/2)**2 <= (.27*min(h,w))**2)
    if modality==1:return (yy>=int(.22*h))&(yy<int(.78*h))
    return (xx>=int(.12*w))&(xx<int(.88*w))&(yy>=int(.10*h))&(yy<int(.90*h))

def apply_perturbation(x,modality,region):
    x=x.clone();mask=torch.from_numpy(roi_mask(modality)).to(x.device)
    if region=="border":mask=~mask
    x[:,modality,:,mask]=0
    return x

def basic(y,p):
    pred=p.argmax(1)
    return {"macro_auc":float(roc_auc_score(y,p,multi_class="ovr",average="macro")),"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro"))}

def binary_counts(y,p,t):
    pred=p>=t;tp=int(((pred)&(y==1)).sum());tn=int(((~pred)&(y==0)).sum());fp=int(((pred)&(y==0)).sum());fn=int(((~pred)&(y==1)).sum())
    div=lambda a,b:float(a/b) if b else None
    return {"threshold":float(t),"sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),"ppv":div(tp,tp+fp),"npv":div(tn,tn+fn),"tp":tp,"tn":tn,"fp":fp,"fn":fn}

def select_threshold(y,p,target,kind):
    candidates=np.unique(np.r_[0,p,1])
    vals=[]
    for t in candidates:
        m=binary_counts(y,p,t); vals.append(m)
    if kind=="specificity":valid=[m for m in vals if m["specificity"]>=target];return max(valid,key=lambda m:(m["sensitivity"],-m["threshold"]))["threshold"]
    valid=[m for m in vals if m["sensitivity"]>=target];return max(valid,key=lambda m:(m["specificity"],m["threshold"]))["threshold"]

def bootstrap_binary(y,p,thresholds,B=5000):
    rng=np.random.default_rng(20260914);values={"auc":[],"auprc":[]}
    for name in thresholds:values[name]={k:[] for k in ("sensitivity","specificity","ppv","npv")}
    for _ in range(B):
        ix=rng.integers(0,len(y),len(y));yy=y[ix];pp=p[ix]
        if len(np.unique(yy))<2:continue
        values["auc"].append(roc_auc_score(yy,pp));values["auprc"].append(average_precision_score(yy,pp))
        for name,t in thresholds.items():
            m=binary_counts(yy,pp,t)
            for k in values[name]:
                if m[k] is not None:values[name][k].append(m[k])
    return {"auc_ci95":np.quantile(values["auc"],[.025,.975]).tolist(),"auprc_ci95":np.quantile(values["auprc"],[.025,.975]).tolist(),"threshold_metric_ci95":{name:{k:np.quantile(v,[.025,.975]).tolist() for k,v in item.items()} for name,item in values.items() if isinstance(item,dict)}}

def bootstrap_stage(y,p,B=5000):
    rng=np.random.default_rng(20260913); pred=p.argmax(1); store={c:{"sensitivity":[],"f1":[],"auprc":[]} for c in range(4)};macro=[]
    for _ in range(B):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix];pp=p[ix];pr=pred[ix]
        if len(np.unique(yy))<4:continue
        aps=[]
        for c in range(4):
            truth=yy==c; pc=pr==c
            sen=(pc&truth).sum()/truth.sum();den=pc.sum()+truth.sum();f=2*(pc&truth).sum()/den if den else 0;ap=average_precision_score(truth,pp[:,c]);aps.append(ap)
            store[c]["sensitivity"].append(sen);store[c]["f1"].append(f);store[c]["auprc"].append(ap)
        macro.append(np.mean(aps))
    out={}
    for c in range(4):
        truth=y==c;pc=pred==c;den=pc.sum()+truth.sum()
        out[str(c)]={"sensitivity":float((pc&truth).sum()/truth.sum()),"sensitivity_ci95":np.quantile(store[c]["sensitivity"],[.025,.975]).tolist(),"f1":float(2*(pc&truth).sum()/den),"f1_ci95":np.quantile(store[c]["f1"],[.025,.975]).tolist(),"auprc":float(average_precision_score(truth,p[:,c])),"auprc_ci95":np.quantile(store[c]["auprc"],[.025,.975]).tolist()}
    return out,float(np.mean([out[str(c)]["auprc"] for c in range(4)])),np.quantile(macro,[.025,.975]).tolist()

def gradcam(model,x,modality,target):
    acts=[];grads=[];module=model.e[modality].layer4[-1]
    h1=module.register_forward_hook(lambda m,i,o:acts.append(o))
    h2=module.register_full_backward_hook(lambda m,gi,go:grads.append(go[0]))
    model.zero_grad(set_to_none=True);logits=forward_masked(model,x);logits[0,target].backward()
    weights=grads[0].mean((2,3),keepdim=True);cam=torch.relu((weights*acts[0]).sum(1,keepdim=True));cam=torch.nn.functional.interpolate(cam,(224,224),mode='bilinear',align_corners=False)[0,0];cam=cam/(cam.max()+1e-8)
    h1.remove();h2.remove();return cam.detach().cpu().numpy()

def main():
    print(torch.cuda.get_device_name(0),flush=True);models=[load_model(s) for s in SEEDS]
    loaders={split:DataLoader(gt.Data(DATA,name,False),batch_size=12,shuffle=False,num_workers=0,pin_memory=True) for split,name in {"val":"valset","test":"testset"}.items()}
    yv,pv,_=predict(models,loaders["val"]);yt,pt,ids=predict(models,loaders["test"])
    stage,macro_ap,macro_ap_ci=bootstrap_stage(yt,pt)
    clinical={"resampling_unit":"eye-level sample","bootstrap_replicates":5000,"stage_names":["normal","early","intermediate","advanced"],"stage_metrics":stage,"macro_auprc":macro_ap,"macro_auprc_ci95":macro_ap_ci,"operating_points":{}}
    for label,classes in {"any_glaucoma_vs_normal":(0,(1,2,3)),"early_vs_normal":(0,(1,))}.items():
        vm=np.isin(yv,(classes[0],*classes[1]));tm=np.isin(yt,(classes[0],*classes[1]));vy=np.isin(yv[vm],classes[1]).astype(int);ty=np.isin(yt[tm],classes[1]).astype(int)
        vp=pv[vm][:,list(classes[1])].sum(1)/(pv[vm][:,classes[0]]+pv[vm][:,list(classes[1])].sum(1));tp=pt[tm][:,list(classes[1])].sum(1)/(pt[tm][:,classes[0]]+pt[tm][:,list(classes[1])].sum(1))
        item={"test_n":int(tm.sum()),"test_auc":float(roc_auc_score(ty,tp)),"test_auprc":float(average_precision_score(ty,tp)),"validation_selected":{}};thresholds={}
        for target in (.90,.95):
            name=f"sensitivity_at_{int(target*100)}pct_specificity";t=select_threshold(vy,vp,target,"specificity");thresholds[name]=t;item["validation_selected"][name]=binary_counts(ty,tp,t)
        for target in (.80,.90):
            name=f"specificity_at_{int(target*100)}pct_sensitivity";t=select_threshold(vy,vp,target,"sensitivity");thresholds[name]=t;item["validation_selected"][name]=binary_counts(ty,tp,t)
        intervals=bootstrap_binary(ty,tp,thresholds)
        item.update({"test_auc_ci95":intervals["auc_ci95"],"test_auprc_ci95":intervals["auprc_ci95"]})
        for name,cis in intervals["threshold_metric_ci95"].items():item["validation_selected"][name]["ci95"]=cis
        clinical["operating_points"][label]=item
    (OUT/'clinical_metrics_extended.json').write_text(json.dumps(clinical,indent=2))
    rows=[]
    combinations={"SLO+OCT+VF":(0,1,2),"SLO+OCT":(0,1),"SLO+VF":(0,2),"OCT+VF":(1,2),"SLO":(0,),"OCT":(1,),"VF":(2,)}
    for name,available in combinations.items():
        y,p,sids=predict(models,loaders['test'],available);row={"available_modalities":name,**basic(y,p)};rows.append(row)
        pd.DataFrame({"id":sids,"label":y,**{f"p{i}":p[:,i] for i in range(4)}}).to_csv(OUT/f"missing_{name.replace('+','_')}_test.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/'missing_modality_performance.csv',index=False)
    base=basic(yt,pt);pert=[]
    for mod,name in enumerate(MODS):
        for region in ('clinical_roi','border'):
            y,p,_=predict(models,loaders['test'],perturb=(mod,region));m=basic(y,p);m.update({"modality":name,"occluded_region":region,"auc_change_from_unperturbed":m['macro_auc']-base['macro_auc'],"mean_true_class_probability_change":float(np.mean(p[np.arange(len(y)),y]-pt[np.arange(len(y)),y]))});pert.append(m)
    pd.DataFrame(pert).to_csv(OUT/'anatomical_perturbation_results.csv',index=False)
    # Stratified Grad-CAM audit on seed 29; overlap is descriptive, not causal validation.
    dataset=loaders['test'].dataset;chosen=[]
    for c in range(4):chosen.extend([i for i in range(len(dataset)) if int(dataset.lab[dataset.ids[i]])==c][:10])
    model=models[1];overlap=[];examples=[]
    for n,i in enumerate(chosen):
        x,y,sid=dataset[i];x=x[None].to(DEVICE)
        for mod,name in enumerate(MODS):
            cam=gradcam(model,x,mod,int(y));roi=roi_mask(mod);overlap.append({"id":sid,"class":int(y),"modality":name,"gradcam_fraction_in_clinical_roi":float(cam[roi].sum()/(cam.sum()+1e-8)),"roi_area_fraction":float(roi.mean())})
            if n in (0,10,20,30): examples.append((n,mod,sid,int(y),x[0,mod].detach().cpu(),cam))
    pd.DataFrame(overlap).to_csv(OUT/'gradcam_roi_overlap.csv',index=False)
    mean=np.array([.485,.456,.406])[:,None,None];std=np.array([.229,.224,.225])[:,None,None]
    fig,axs=plt.subplots(4,3,figsize=(9,10))
    for n,mod,sid,y,img,cam in examples:
        row=(0,10,20,30).index(n);arr=np.clip(img.numpy()*std+mean,0,1).transpose(1,2,0);axs[row,mod].imshow(arr);axs[row,mod].imshow(cam,cmap='jet',alpha=.38);axs[row,mod].axis('off');axs[row,mod].set_title(f"{MODS[mod]} | {['N','E','I','A'][y]} | {sid}",fontsize=9)
    fig.suptitle('Grad-CAM examples (seed 29; descriptive only)',fontsize=13);fig.tight_layout();fig.savefig(OUT/'gradcam_examples.png',dpi=300);plt.close(fig)
    print(json.dumps(clinical,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(pert).to_string(index=False))

if __name__=='__main__':main()
