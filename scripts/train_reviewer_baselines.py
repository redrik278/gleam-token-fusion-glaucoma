"""Reviewer-requested independently trained modality addition and simple fusion baselines."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score,f1_score,roc_auc_score
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_gleam_fourclass import Data,resnet,seed_all,met,predict

class SubsetNet(nn.Module):
 def __init__(self,idx):
  super().__init__(); self.idx=idx; self.e=nn.ModuleList([resnet() for _ in idx]); self.h=nn.Sequential(nn.Linear(512*len(idx),256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,4))
 def forward(self,x,drop=False): return self.h(torch.cat([e(x[:,j]) for e,j in zip(self.e,self.idx)],1)),None
class WeightedLogits(nn.Module):
 def __init__(self):
  super().__init__();self.e=nn.ModuleList([resnet() for _ in range(3)]);self.h=nn.ModuleList([nn.Linear(512,4) for _ in range(3)]);self.w=nn.Parameter(torch.zeros(3))
 def forward(self,x,drop=False):
  z=torch.stack([h(e(x[:,i])) for i,(e,h) in enumerate(zip(self.e,self.h))],1);return (z*torch.softmax(self.w,0)[None,:,None]).sum(1),None
class AttentionPool(nn.Module):
 def __init__(self,d=256):
  super().__init__();self.e=nn.ModuleList([resnet() for _ in range(3)]);self.p=nn.ModuleList([nn.Sequential(nn.Linear(512,d),nn.LayerNorm(d)) for _ in range(3)]);self.score=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1));self.h=nn.Linear(d,4)
 def forward(self,x,drop=False):
  t=torch.stack([p(e(x[:,i])) for i,(e,p) in enumerate(zip(self.e,self.p))],1);a=torch.softmax(self.score(t),1);return self.h((a*t).sum(1)),None
class MLPMatched(nn.Module):
 def __init__(self,d=256):
  super().__init__();self.e=nn.ModuleList([resnet() for _ in range(3)]);self.p=nn.ModuleList([nn.Sequential(nn.Linear(512,d),nn.LayerNorm(d)) for _ in range(3)]);self.f=nn.Sequential(nn.Linear(3*d,512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,256),nn.GELU(),nn.Dropout(.2));self.h=nn.Linear(256,4)
 def forward(self,x,drop=False):
  t=torch.cat([p(e(x[:,i])) for i,(e,p) in enumerate(zip(self.e,self.p))],1);return self.h(self.f(t)),None
class TokenSeparated(nn.Module):
 def __init__(self,use_aux=True,use_md=True,d=256):
  super().__init__();self.use_aux=use_aux;self.use_md=use_md;self.e=nn.ModuleList([resnet() for _ in range(3)]);self.p=nn.ModuleList([nn.Sequential(nn.Linear(512,d),nn.LayerNorm(d)) for _ in range(3)]);self.g=nn.ModuleList([nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid()) for _ in range(3)]);self.mod=nn.Parameter(torch.randn(1,3,d)*.02);self.cls=nn.Parameter(torch.randn(1,1,d)*.02);lay=nn.TransformerEncoderLayer(d,8,d*4,.2,batch_first=True,norm_first=True,activation='gelu');self.f=nn.TransformerEncoder(lay,2);self.h=nn.Linear(d,4);self.aux=nn.ModuleList([nn.Linear(d,4) for _ in range(3)])
 def forward(self,x,drop=False):
  raw=[e(x[:,i]) for i,e in enumerate(self.e)];t=torch.stack([p(z)*g(z) for p,g,z in zip(self.p,self.g,raw)],1)+self.mod
  if self.training and drop and self.use_md:
   q=torch.rand(t.shape[:2],device=t.device)<.15;q[q.all(1),0]=False;t=t.masked_fill(q[...,None],0)
  o=self.f(torch.cat([self.cls.expand(x.size(0),-1,-1),t],1))[:,0]
  return self.h(o),(torch.stack([h(t[:,i]) for i,h in enumerate(self.aux)],1) if self.use_aux else None)
def make(k):
 return {'vf':lambda:SubsetNet([2]),'vf_oct':lambda:SubsetNet([2,1]),'vf_oct_slo':lambda:SubsetNet([2,1,0]),'weighted_logits':WeightedLogits,'attention_pool':AttentionPool,'mlp_matched':MLPMatched,'token_no_aux':lambda:TokenSeparated(False,True),'token_no_moddrop':lambda:TokenSeparated(True,False)}[k]()
def fit(k,s,a,L,dev):
 done=a.out/f'{k}_seed{s}_result.json'
 if done.exists(): print('SKIP',done,flush=True);return json.loads(done.read_text())
 seed_all(s);torch.cuda.reset_peak_memory_stats();m=make(k).to(dev);params=sum(p.numel() for p in m.parameters());opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);sc=torch.amp.GradScaler('cuda');ce=nn.CrossEntropyLoss();best=-1;stale=0;ck=a.out/f'{k}_seed{s}.pt';hist=[];start=time.time()
 for ep in range(1,a.epochs+1):
  m.train();ls=[]
  for x,y,_ in L['train']:
   x=x.to(dev);y=y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda'):
    z,aux=m(x,True);loss=ce(z,y)+(0.15*sum(ce(aux[:,i],y) for i in range(3))/3 if aux is not None else 0)
   sc.scale(loss).backward();sc.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),5);sc.step(opt);sc.update();ls.append(loss.item())
  sch.step();yv,pv,_=predict(m,L['val'],dev);v=met(yv,pv);hist.append({'epoch':ep,'loss':float(np.mean(ls)),**v});print(k,s,ep,round(v['auc_macro_ovr'],4),flush=True)
  if v['auc_macro_ovr']>best+1e-5:best=v['auc_macro_ovr'];stale=0;torch.save({'model':m.state_dict(),'epoch':ep,'val':v},ck)
  else:stale+=1
  if stale>=5:break
 st=torch.load(ck,map_location=dev,weights_only=True);m.load_state_dict(st['model']);yt,pt,ids=predict(m,L['test'],dev);t=met(yt,pt);pd.DataFrame({'id':ids,'label':yt,**{f'p{i}':pt[:,i] for i in range(4)}}).to_csv(a.out/f'{k}_seed{s}_test.csv',index=False);r={'model':k,'seed':s,'best_epoch':st['epoch'],'validation':st['val'],'test':t,'parameters':params,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20,'training_seconds':time.time()-start};done.write_text(json.dumps(r,indent=2));(a.out/f'{k}_seed{s}_history.json').write_text(json.dumps(hist,indent=2));print('TEST',k,s,t,flush=True);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/public/GLEAM'));p.add_argument('--out',type=Path,default=Path('output/gleam_fourclass'));p.add_argument('--models',nargs='+',default=['vf','vf_oct','vf_oct_slo','weighted_logits','attention_pool','mlp_matched','token_no_aux','token_no_moddrop']);p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=12);p.add_argument('--lr',type=float,default=1e-4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda');print(torch.cuda.get_device_name(0),flush=True);ds={q:Data(a.data,s,q=='train') for q,s in {'train':'trainset','val':'valset','test':'testset'}.items()};L={q:DataLoader(d,batch_size=a.batch,shuffle=q=='train',num_workers=0,pin_memory=True) for q,d in ds.items()};R=[]
 for k in a.models:
  for s in a.seeds:R.append(fit(k,s,a,L,dev));torch.cuda.empty_cache()
 (a.out/'reviewer_baseline_results.json').write_text(json.dumps(R,indent=2))
if __name__=='__main__':main()
