"""Locked patient-independent Harvard-GDP experiment (protocol 2026-09-16)."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,f1_score,brier_score_loss,log_loss
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms

SEEDS=(17,29,43,71,101)
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=True
class GDP(Dataset):
 def __init__(self,root,frame,train=False):
  self.root=Path(root);self.d=frame.reset_index(drop=True);self.train=train
  self.tf=transforms.Compose(([transforms.RandomResizedCrop(224,scale=(.9,1.0))] if train else [transforms.Resize((224,224))]))
  self.td=self.d[[c for c in self.d if c.startswith('td')]].to_numpy(np.float32)
  self.td=np.nan_to_num(self.td,nan=-35,posinf=5,neginf=-35);self.td=np.clip(self.td,-40,10)/25.0
 def __len__(self):return len(self.d)
 def __getitem__(self,i):
  r=self.d.iloc[i];p=self.root/'RNFLT'/(str(r.filename)+'.npz')
  z=np.load(p);a=np.asarray(z['rnflt'],dtype=np.float32);a=np.nan_to_num(a);a=np.clip(a,0,200)/100-1
  if a.ndim==3:a=a.squeeze()
  x=torch.from_numpy(a)[None];x=self.tf(x);return x,torch.from_numpy(self.td[i]),torch.tensor(float(r.glaucoma)),str(r.filename)
def octenc():
 m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);w=m.conv1.weight.data.mean(1,keepdim=True);m.conv1=nn.Conv2d(1,64,7,2,3,bias=False);m.conv1.weight.data.copy_(w);m.fc=nn.Identity();return m
class OCT(nn.Module):
 def __init__(self):super().__init__();self.o=octenc();self.h=nn.Linear(512,1)
 def forward(self,x,v,drop=False):return self.h(self.o(x)).squeeze(1),{}
class VF(nn.Module):
 def __init__(self):super().__init__();self.v=nn.Sequential(nn.Linear(52,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,128),nn.GELU());self.h=nn.Linear(128,1)
 def forward(self,x,v,drop=False):return self.h(self.v(v)).squeeze(1),{}
class Late(nn.Module):
 def __init__(self):super().__init__();self.o=octenc();self.v=nn.Sequential(nn.Linear(52,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,128),nn.GELU());self.h=nn.Sequential(nn.Linear(640,256),nn.GELU(),nn.Dropout(.25),nn.Linear(256,1))
 def forward(self,x,v,drop=False):return self.h(torch.cat([self.o(x),self.v(v)],1)).squeeze(1),{}
class Reliability(nn.Module):
 def __init__(self):
  super().__init__();self.o=octenc();self.v=nn.Sequential(nn.Linear(52,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,128),nn.GELU());self.op=nn.Sequential(nn.Linear(512,128),nn.LayerNorm(128),nn.GELU());self.ol=nn.Linear(128,1);self.vl=nn.Linear(128,1);self.ov=nn.Linear(128,1);self.vv=nn.Linear(128,1);self.r=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
 def forward(self,x,v,drop=False):
  a=self.op(self.o(x));b=self.v(v);lo=self.ol(a).squeeze(1);lv=self.vl(b).squeeze(1);u=torch.stack([self.ov(a).squeeze(1),self.vv(b).squeeze(1)],1).clamp(-4,4);w=torch.softmax(-u,1)
  if self.training and drop:
   q=torch.rand(len(x),device=x.device);oct_missing=(q<.1)[:,None];vf_missing=((q>=.1)&(q<.2))[:,None]
   w=torch.where(oct_missing,torch.tensor([[0.,1.]],device=x.device),w)
   w=torch.where(vf_missing,torch.tensor([[1.,0.]],device=x.device),w)
  z=w[:,0]*lo+w[:,1]*lv+self.r(torch.cat([a,b],1)).squeeze(1)
  return z,{'lo':lo,'lv':lv,'u':u,'w':w}
def make(k):return {'oct':OCT,'vf':VF,'late':Late,'reliability':Reliability}[k]()
@torch.inference_mode()
def pred(m,L,dev):
 m.eval();Y=[];P=[];I=[];W=[]
 for x,v,y,i in L:
  with torch.autocast('cuda'):z,d=m(x.to(dev),v.to(dev));p=z.sigmoid()
  Y+=y.numpy().tolist();P+=p.float().cpu().numpy().tolist();I+=list(i)
  if 'w' in d:W+=d['w'].float().cpu().numpy().tolist()
 return np.array(Y),np.array(P),I,np.array(W)
def met(y,p,t=.5):
 q=p>=t;return {'auc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'f1':float(f1_score(y,q)),'brier':float(brier_score_loss(y,p)),'nll':float(log_loss(y,np.c_[1-p,p]))}
def fit(k,s,a,L,dev):
 done=a.out/f'{k}_seed{s}_result.json'
 if done.exists():return json.loads(done.read_text())
 seedall(s);m=make(k).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);sc=torch.amp.GradScaler('cuda');ce=nn.BCEWithLogitsLoss();best=-1;stale=0;ck=a.out/f'{k}_seed{s}.pt';hist=[];start=time.time()
 for ep in range(1,a.epochs+1):
  m.train();ls=[]
  for x,v,y,_ in L['train']:
   x=x.to(dev);v=v.to(dev);y=y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda'):
    z,d=m(x,v,k=='reliability');loss=ce(z,y)
    if k=='reliability':
     blo=nn.functional.binary_cross_entropy_with_logits(d['lo'],y,reduction='none');blv=nn.functional.binary_cross_entropy_with_logits(d['lv'],y,reduction='none');loss+=.15*(blo.mean()+blv.mean())+.03*((torch.exp(-d['u'][:,0])*blo+d['u'][:,0]).mean()+(torch.exp(-d['u'][:,1])*blv+d['u'][:,1]).mean())
   sc.scale(loss).backward();sc.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),5);sc.step(opt);sc.update();ls.append(loss.item())
  sch.step();yv,pv,_,_=pred(m,L['val'],dev);z=met(yv,pv);hist.append({'epoch':ep,'loss':float(np.mean(ls)),**z});print(k,s,ep,round(z['auc'],4),flush=True)
  if z['auc']>best+1e-5:best=z['auc'];stale=0;torch.save({'model':m.state_dict(),'epoch':ep,'val':z},ck)
  else:stale+=1
  if stale>=6:break
 st=torch.load(ck,map_location=dev,weights_only=True);m.load_state_dict(st['model']);
 for split in ('val','test'):
  y,p,ids,w=pred(m,L[split],dev);pd.DataFrame({'id':ids,'label':y,'probability':p}).to_csv(a.out/f'{k}_seed{s}_{split}.csv',index=False)
  if len(w):pd.DataFrame({'id':ids,'oct_weight':w[:,0],'vf_weight':w[:,1]}).to_csv(a.out/f'{k}_seed{s}_{split}_weights.csv',index=False)
 r={'model':k,'seed':s,'best_epoch':st['epoch'],'validation':st['val'],'parameters':sum(p.numel() for p in m.parameters()),'training_seconds':time.time()-start};done.write_text(json.dumps(r,indent=2));return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/public/Harvard-GDP/extracted'));p.add_argument('--meta',type=Path,default=Path('data/public/Harvard-GDP/data_summary.csv'));p.add_argument('--out',type=Path,default=Path('output/harvard_gdp_locked'));p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.meta);tr=d[d.glaucoma_detection_use=='training'];te=d[d.glaucoma_detection_use=='test'];devf,valf=train_test_split(tr,test_size=.2,random_state=20260916,stratify=tr.glaucoma)
 (a.out/'split.csv').write_text(pd.concat([devf.assign(split='development'),valf.assign(split='validation'),te.assign(split='test')])[['filename','glaucoma','split']].to_csv(index=False))
 sets={'train':GDP(a.data,devf,True),'val':GDP(a.data,valf),'test':GDP(a.data,te)};L={k:DataLoader(v,batch_size=a.batch,shuffle=k=='train',num_workers=0,pin_memory=True) for k,v in sets.items()};device='cuda';print(torch.cuda.get_device_name(0),{k:len(v) for k,v in sets.items()},flush=True)
 for k in ('oct','vf','late','reliability'):
  for s in SEEDS:fit(k,s,a,L,device);torch.cuda.empty_cache()
if __name__=='__main__':main()
