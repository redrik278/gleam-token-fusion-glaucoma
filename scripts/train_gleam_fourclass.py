"""Repeated-seed four-class GLEAM benchmark, including official HAMM architecture."""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score,f1_score,roc_auc_score
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"external"/"HAMM"))
from Model import HAMM

NAMES=("SLO.jpg","Thickness.jpg","VF.jpg")
def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=True
class Data(Dataset):
 def __init__(self,root,split,train=False):
  self.root=Path(root)/"split"/split; fn={"trainset":"trainsetlabel.csv","valset":"valsetlabel.csv","testset":"testsetlabel.csv"}[split]
  d=pd.read_csv(Path(root)/"split"/fn);self.lab={f"{int(r.ID):04d}":int(r.Label) for _,r in d.iterrows()};self.ids=sorted(self.lab)
  norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
  self.tf=transforms.Compose(([transforms.RandomResizedCrop(224,scale=(.85,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.1,.1,.05,.02)] if train else [transforms.Resize((224,224))])+[transforms.ConvertImageDtype(torch.float32),norm])
  # Cache compact uint8 tensors in RAM once; source JPEGs live on an HDD.
  self.cache={}
  for sid in self.ids:
   imgs=[]
   for n in NAMES:
    im=Image.open(self.root/sid/n).convert("RGB").resize((256,256))
    imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1))
   self.cache[sid]=imgs
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  sid=self.ids[i];return torch.stack([self.tf(im) for im in self.cache[sid]]),torch.tensor(self.lab[sid]),sid
def resnet():m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);m.fc=nn.Identity();return m
class Early(nn.Module):
 def __init__(self):
  super().__init__();self.net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);w=self.net.conv1.weight.data;self.net.conv1=nn.Conv2d(9,64,7,2,3,bias=False);self.net.conv1.weight.data.copy_(w.repeat(1,3,1,1)/3);self.net.fc=nn.Linear(512,4)
 def forward(self,x,drop=False):return self.net(x.flatten(1,2)),None
class Late(nn.Module):
 def __init__(self):super().__init__();self.e=nn.ModuleList([resnet() for _ in range(3)]);self.h=nn.Sequential(nn.Linear(1536,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,4))
 def forward(self,x,drop=False):return self.h(torch.cat([e(x[:,i]) for i,e in enumerate(self.e)],1)),None
class Token(nn.Module):
 def __init__(self,dim=256,use_gates=True,use_transformer=True,use_regularizers=True):
  super().__init__();self.use_gates=use_gates;self.use_transformer=use_transformer;self.use_regularizers=use_regularizers;self.e=nn.ModuleList([resnet() for _ in range(3)]);self.p=nn.ModuleList([nn.Sequential(nn.Linear(512,dim),nn.LayerNorm(dim)) for _ in range(3)]);self.g=nn.ModuleList([nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid()) for _ in range(3)]);self.mod=nn.Parameter(torch.randn(1,3,dim)*.02);self.cls=nn.Parameter(torch.randn(1,1,dim)*.02);lay=nn.TransformerEncoderLayer(dim,8,dim*4,.2,batch_first=True,norm_first=True,activation="gelu");self.f=nn.TransformerEncoder(lay,2);self.h=nn.Linear(dim,4);self.aux=nn.ModuleList([nn.Linear(dim,4) for _ in range(3)])
 def forward(self,x,drop=False):
  raw=[e(x[:,i]) for i,e in enumerate(self.e)];t=torch.stack([p(z)*(g(z) if self.use_gates else 1) for p,g,z in zip(self.p,self.g,raw)],1)+self.mod
  if self.training and drop and self.use_regularizers:
   q=torch.rand(t.shape[:2],device=t.device)<.15;q[q.all(1),0]=False;t=t.masked_fill(q[...,None],0)
  if self.use_transformer:o=self.f(torch.cat([self.cls.expand(x.size(0),-1,-1),t],1))[:,0]
  else:o=t.mean(1)
  return self.h(o),(torch.stack([h(t[:,i]) for i,h in enumerate(self.aux)],1) if self.use_regularizers else None)
class Official(nn.Module):
 def __init__(self):super().__init__();self.net=HAMM(num_classes=4,att=True)
 def forward(self,x,drop=False):return self.net(x[:,0],x[:,1],x[:,2]),None
def make(k):
 if k=="token_no_gates":return Token(use_gates=False)
 if k=="token_no_transformer":return Token(use_transformer=False)
 if k=="token_no_regularizers":return Token(use_regularizers=False)
 return {"early_fusion":Early,"late_fusion":Late,"gated_token":Token,"official_hamm":Official}[k]()
def met(y,p):
 pred=p.argmax(1);one=np.eye(4)[y];ece=0
 conf=p.max(1);ok=pred==y
 for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
  q=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9));ece+=q.mean()*abs(ok[q].mean()-conf[q].mean()) if q.any() else 0
 return {"auc_macro_ovr":float(roc_auc_score(y,p,multi_class="ovr",average="macro")),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"f1_macro":float(f1_score(y,pred,average="macro")),"quadratic_kappa":float(cohen_kappa_score(y,pred,weights="quadratic")),"nll":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-7,1)).mean()),"brier_multiclass":float(((p-one)**2).sum(1).mean()),"ece10":float(ece)}
@torch.inference_mode()
def predict(m,l,dev):
 m.eval();Y=[];P=[];I=[]
 for x,y,i in l:
  with torch.autocast("cuda"):z,_=m(x.to(dev));p=z.softmax(1)
  Y.extend(y.numpy());P.extend(p.float().cpu().numpy());I.extend(i)
 return np.array(Y),np.array(P),I
def fit(k,s,a,L,dev):
 out=a.out;done=out/f"{k}_seed{s}_result.json"
 if done.exists():print("SKIP",done);return json.loads(done.read_text())
 seed_all(s);torch.cuda.reset_peak_memory_stats();m=make(k).to(dev);params=sum(x.numel() for x in m.parameters());opt=torch.optim.AdamW(m.parameters(),lr=(3e-6 if k=="official_hamm" else a.lr),weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);sc=torch.amp.GradScaler("cuda");ce=nn.CrossEntropyLoss();best=-1;stale=0;ck=out/f"{k}_seed{s}.pt";hist=[];start=time.time()
 for ep in range(1,a.epochs+1):
  m.train();ls=[]
  for x,y,_ in L["train"]:
   x=x.to(dev);y=y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast("cuda"):z,aux=m(x,k.startswith("gated_token") or k.startswith("token_"));loss=ce(z,y)+(0.15*sum(ce(aux[:,i],y) for i in range(3))/3 if aux is not None else 0)
   sc.scale(loss).backward();sc.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),5);sc.step(opt);sc.update();ls.append(loss.item())
  sch.step();yv,pv,_=predict(m,L["val"],dev);v=met(yv,pv);hist.append({"epoch":ep,"loss":float(np.mean(ls)),**v});print(k,s,ep,round(v["auc_macro_ovr"],4),flush=True)
  if v["auc_macro_ovr"]>best+1e-5:best=v["auc_macro_ovr"];stale=0;torch.save({"model":m.state_dict(),"epoch":ep,"val":v},ck)
  else: stale+=1
  if stale>=5: print(k,s,"early_stop",ep,flush=True);break
 st=torch.load(ck,map_location=dev,weights_only=True);m.load_state_dict(st["model"]);yt,pt,ids=predict(m,L["test"],dev);t=met(yt,pt)
 pd.DataFrame({"id":ids,"label":yt,**{f"p{i}":pt[:,i] for i in range(4)}}).to_csv(out/f"{k}_seed{s}_test.csv",index=False)
 r={"model":k,"seed":s,"best_epoch":st["epoch"],"validation":st["val"],"test":t,"parameters":params,"peak_gpu_mb":torch.cuda.max_memory_allocated()/2**20,"training_seconds":time.time()-start};done.write_text(json.dumps(r,indent=2));(out/f"{k}_seed{s}_history.json").write_text(json.dumps(hist,indent=2));print("TEST",json.dumps(r),flush=True);return r
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data/public/GLEAM"));p.add_argument("--out",type=Path,default=Path("output/gleam_fourclass"));p.add_argument("--models",nargs="+",default=["early_fusion","late_fusion","gated_token","official_hamm"]);p.add_argument("--seeds",nargs="+",type=int,default=[17,29,43]);p.add_argument("--epochs",type=int,default=12);p.add_argument("--batch",type=int,default=12);p.add_argument("--lr",type=float,default=1e-4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);dev=torch.device("cuda");print(torch.cuda.get_device_name(0),flush=True)
 ds={q:Data(a.data,s,q=="train") for q,s in {"train":"trainset","val":"valset","test":"testset"}.items()};L={q:DataLoader(d,batch_size=(2 if "official_hamm" in a.models else a.batch),shuffle=q=="train",num_workers=0,pin_memory=True) for q,d in ds.items()}
 # Recreate loaders per model because HAMM needs a smaller batch on 8 GB.
 R=[]
 for k in a.models:
  Lk={q:DataLoader(d,batch_size=(2 if k=="official_hamm" else a.batch),shuffle=q=="train",num_workers=0,pin_memory=True) for q,d in ds.items()}
  for s in a.seeds:R.append(fit(k,s,a,Lk,dev));torch.cuda.empty_cache()
 (a.out/"results.json").write_text(json.dumps(R,indent=2));pd.json_normalize(R).to_csv(a.out/"results.csv",index=False)
if __name__=="__main__":main()
