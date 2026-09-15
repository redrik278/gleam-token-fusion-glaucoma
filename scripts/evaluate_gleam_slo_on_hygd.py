"""Frozen GLEAM SLO auxiliary-head transfer to patient-grouped HYGD."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_gleam_fourclass import Token
ROOT=Path('data/public/HYGD/1.1.0');OUT=Path('output/hygd_frozen_transfer');OUT.mkdir(parents=True,exist_ok=True)
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
d=pd.read_csv(ROOT/'Labels.csv');d['y']=(d.Label=='GON+').astype(int);dev='cuda';models=[]
for s in (17,29,43):
 m=Token().to(dev);m.load_state_dict(torch.load(f'output/gleam_fourclass/gated_token_seed{s}.pt',map_location=dev,weights_only=True)['model']);m.eval();models.append(m)
P=[]
with torch.inference_mode():
 for start in range(0,len(d),24):
  ims=torch.stack([tf(Image.open(ROOT/'Images'/n).convert('RGB')) for n in d['Image Name'].iloc[start:start+24]]).to(dev);ps=[]
  for m in models:
   with torch.autocast('cuda'):
    raw=m.e[0](ims);t=m.p[0](raw)*m.g[0](raw)+m.mod[:,0];z=m.aux[0](t);prob=1-z.softmax(1)[:,0]
   ps.append(prob.float().cpu().numpy())
  P.extend(np.mean(ps,0))
d['probability']=P;d.to_csv(OUT/'image_predictions.csv',index=False)
p=d.groupby('Patient').agg(label=('y','first'),probability=('probability','mean'),images=('y','size'),quality=('Quality Score','mean')).reset_index();p.to_csv(OUT/'patient_predictions.csv',index=False)
y=p.label.to_numpy();pr=p.probability.to_numpy();q=pr>=.5;tn,fp,fn,tp=confusion_matrix(y,q).ravel();res={'patients':len(p),'images':len(d),'positive_patients':int(y.sum()),'negative_patients':int((1-y).sum()),'auc':float(roc_auc_score(y,pr)),'auprc':float(average_precision_score(y,pr)),'accuracy':float(accuracy_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'f1':float(f1_score(y,q)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp))}
rng=np.random.default_rng(20260916);vals=[]
for _ in range(10000):
 ix=rng.integers(0,len(y),len(y))
 if len(np.unique(y[ix]))==2:vals.append(roc_auc_score(y[ix],pr[ix]))
res['auc_ci95']=[float(x) for x in np.quantile(vals,[.025,.975])];(OUT/'results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
