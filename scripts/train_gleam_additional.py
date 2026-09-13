"""Additional prespecified-style sensitivity experiments requested during review.

Runs (1) modality-specific augmentation with the original gated token model and
(2) a cumulative-link ordinal token model on the unchanged public GLEAM split.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

NAMES = ("SLO.jpg", "Thickness.jpg", "VF.jpg")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ClinicalAugData(Dataset):
    """Augment SLO appearance only; preserve OCT/VF geometry and intensities."""
    def __init__(self, root, split, train=False):
        self.root = Path(root) / "split" / split
        fn = {"trainset":"trainsetlabel.csv", "valset":"valsetlabel.csv", "testset":"testsetlabel.csv"}[split]
        frame = pd.read_csv(Path(root) / "split" / fn)
        self.labels = {f"{int(r.ID):04d}": int(r.Label) for _, r in frame.iterrows()}
        self.ids = sorted(self.labels)
        norm = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        finish = [transforms.ConvertImageDtype(torch.float32), norm]
        self.slo = transforms.Compose(([
            transforms.RandomResizedCrop(224, scale=(.90,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.1,.1,.05,.02),
        ] if train else [transforms.Resize((224,224))]) + finish)
        self.fixed = transforms.Compose([transforms.Resize((224,224)), *finish])
        self.cache = {}
        for sid in self.ids:
            self.cache[sid] = [torch.from_numpy(np.asarray(Image.open(self.root/sid/name).convert("RGB").resize((256,256))).copy()).permute(2,0,1) for name in NAMES]

    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        sid = self.ids[index]; images = self.cache[sid]
        return torch.stack([self.slo(images[0]), self.fixed(images[1]), self.fixed(images[2])]), torch.tensor(self.labels[sid]), sid


class SharedAugData(ClinicalAugData):
    """Reproduce the original common augmentation for the ordinal comparison."""
    def __init__(self, root, split, train=False):
        super().__init__(root,split,False)
        norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        self.shared=transforms.Compose(([
            transforms.RandomResizedCrop(224,scale=(.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.1,.1,.05,.02),
        ] if train else [transforms.Resize((224,224))])+[transforms.ConvertImageDtype(torch.float32),norm])
    def __getitem__(self,index):
        sid=self.ids[index]
        return torch.stack([self.shared(image) for image in self.cache[sid]]),torch.tensor(self.labels[sid]),sid


def resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); model.fc = nn.Identity(); return model


class TokenBackbone(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoders = nn.ModuleList([resnet() for _ in range(3)])
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(512,dim),nn.LayerNorm(dim)) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid()) for _ in range(3)])
        self.modality_embeddings = nn.Parameter(torch.randn(1,3,dim)*.02)
        self.cls = nn.Parameter(torch.randn(1,1,dim)*.02)
        layer = nn.TransformerEncoderLayer(dim,8,dim*4,.2,batch_first=True,norm_first=True,activation="gelu")
        self.transformer = nn.TransformerEncoder(layer,2)

    def features(self, x, dropout=False):
        raw = [encoder(x[:,i]) for i,encoder in enumerate(self.encoders)]
        tokens = torch.stack([proj(value)*gate(value) for proj,gate,value in zip(self.projections,self.gates,raw)],1) + self.modality_embeddings
        if self.training and dropout:
            mask = torch.rand(tokens.shape[:2], device=tokens.device) < .15
            mask[mask.all(1),0] = False
            tokens = tokens.masked_fill(mask[...,None],0)
        fused = self.transformer(torch.cat([self.cls.expand(x.size(0),-1,-1),tokens],1))[:,0]
        return fused, tokens


class GatedClinicalAug(TokenBackbone):
    def __init__(self):
        super().__init__(); self.head=nn.Linear(256,4); self.aux=nn.ModuleList([nn.Linear(256,4) for _ in range(3)])
    def forward(self,x,dropout=False):
        fused,tokens=self.features(x,dropout); return self.head(fused),torch.stack([h(tokens[:,i]) for i,h in enumerate(self.aux)],1)


class OrdinalToken(TokenBackbone):
    def __init__(self):
        super().__init__(); self.score=nn.Linear(256,1); self.aux=nn.ModuleList([nn.Linear(256,1) for _ in range(3)]); self.raw_steps=nn.Parameter(torch.zeros(3))
    def thresholds(self):
        values=torch.cumsum(F.softplus(self.raw_steps),0); return values-values.mean()
    def forward(self,x,dropout=False):
        fused,tokens=self.features(x,dropout); cuts=self.thresholds()
        logits=self.score(fused)-cuts[None,:]
        aux=torch.stack([h(tokens[:,i]).squeeze(1) for i,h in enumerate(self.aux)],1)[...,None]-cuts[None,None,:]
        return logits,aux


def ordinal_prob(logits):
    q=torch.sigmoid(logits)
    return torch.stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]],1).clamp_min(0).div(torch.stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]],1).clamp_min(0).sum(1,keepdim=True))


def metrics(y,p):
    pred=p.argmax(1)
    return {"auc_macro_ovr":float(roc_auc_score(y,p,multi_class="ovr",average="macro")),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"f1_macro":float(f1_score(y,pred,average="macro")),"quadratic_kappa":float(cohen_kappa_score(y,pred,weights="quadratic"))}


@torch.inference_mode()
def predict(model,loader,device,ordinal):
    model.eval(); ys=[]; ps=[]; ids=[]
    for x,y,sid in loader:
        with torch.autocast("cuda"):
            logits,_=model(x.to(device)); p=ordinal_prob(logits) if ordinal else logits.softmax(1)
        ys.extend(y.numpy()); ps.extend(p.float().cpu().numpy()); ids.extend(sid)
    return np.asarray(ys),np.asarray(ps),ids


def fit(kind,seed,args,loaders,device):
    out=args.out; result_path=out/f"{kind}_seed{seed}_result.json"
    if result_path.exists(): print("SKIP",result_path,flush=True); return
    seed_all(seed); torch.cuda.reset_peak_memory_stats()
    ordinal=kind=="ordinal_token"; model=(OrdinalToken() if ordinal else GatedClinicalAug()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    scaler=torch.amp.GradScaler("cuda"); ce=nn.CrossEntropyLoss(); bce=nn.BCEWithLogitsLoss()
    best=-1.; stale=0; history=[]; checkpoint=out/f"{kind}_seed{seed}.pt"; start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y,_ in loaders["train"]:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits,aux=model(x,True)
                if ordinal:
                    target=(y[:,None]>torch.arange(3,device=device)[None,:]).float()
                    loss=bce(logits,target)+.15*sum(bce(aux[:,i],target) for i in range(3))/3
                else:
                    loss=ce(logits,y)+.15*sum(ce(aux[:,i],y) for i in range(3))/3
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),5); scaler.step(optimizer); scaler.update(); losses.append(loss.item())
        scheduler.step(); yv,pv,_=predict(model,loaders["val"],device,ordinal); val=metrics(yv,pv); history.append({"epoch":epoch,"loss":float(np.mean(losses)),**val}); print(kind,seed,epoch,round(val["auc_macro_ovr"],4),flush=True)
        if val["auc_macro_ovr"]>best+1e-5:
            best=val["auc_macro_ovr"]; stale=0; torch.save({"model":model.state_dict(),"epoch":epoch,"val":val},checkpoint)
        else: stale+=1
        if stale>=5: print(kind,seed,"early_stop",epoch,flush=True); break
    state=torch.load(checkpoint,map_location=device,weights_only=True); model.load_state_dict(state["model"]); yt,pt,ids=predict(model,loaders["test"],device,ordinal); test=metrics(yt,pt)
    pd.DataFrame({"id":ids,"label":yt,**{f"p{i}":pt[:,i] for i in range(4)}}).to_csv(out/f"{kind}_seed{seed}_test.csv",index=False)
    result={"model":kind,"seed":seed,"best_epoch":state["epoch"],"validation":state["val"],"test":test,"parameters":sum(p.numel() for p in model.parameters()),"peak_gpu_mb":torch.cuda.max_memory_allocated()/2**20,"training_seconds":time.time()-start,"augmentation":"SLO-only appearance augmentation; deterministic OCT and VF"}
    result_path.write_text(json.dumps(result,indent=2)); (out/f"{kind}_seed{seed}_history.json").write_text(json.dumps(history,indent=2)); print("TEST",json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=Path('data/public/GLEAM')); parser.add_argument('--out',type=Path,default=Path('output/gleam_fourclass')); parser.add_argument('--models',nargs='+',default=['gated_token_clinical_aug','ordinal_token']); parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]); parser.add_argument('--epochs',type=int,default=12); parser.add_argument('--batch',type=int,default=12); parser.add_argument('--lr',type=float,default=1e-4); args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda'); print(torch.cuda.get_device_name(0),flush=True)
    split_names={'train':'trainset','val':'valset','test':'testset'}
    clinical={key:ClinicalAugData(args.data,name,key=='train') for key,name in split_names.items()}
    shared={key:SharedAugData(args.data,name,key=='train') for key,name in split_names.items()}
    for kind in args.models:
        datasets=shared if kind=='ordinal_token' else clinical
        loaders={key:DataLoader(ds,batch_size=args.batch,shuffle=key=='train',num_workers=0,pin_memory=True) for key,ds in datasets.items()}
        for seed in args.seeds: fit(kind,seed,args,loaders,device); torch.cuda.empty_cache()

if __name__=='__main__': main()
