from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0,str(Path.cwd()))
from scripts.train_gleam_fourclass import Token
M=Path('data/processed_v2/model_manifest_structural_functional_definite.csv');O=Path('output/gleam_fourclass');d=pd.read_csv(M)
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]);tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm]);dev='cuda';models=[]
for s in [17,29,43]:
 m=Token().to(dev);m.load_state_dict(torch.load(O/f'gated_token_seed{s}.pt',map_location=dev,weights_only=True)['model']);m.eval();models.append(m)
rows=[]
for _,r in d.iterrows():
 paths=[r['fundus_input'],r['oct_input'],r['visual_field_input']]
 if not all(Path(p).exists() for p in paths):continue
 x=torch.stack([tf(Image.open(p).convert('RGB')) for p in paths])[None].to(dev);ps=[]
 with torch.inference_mode(),torch.autocast('cuda'):
  for m in models:ps.append(m(x)[0].softmax(1).float().cpu().numpy()[0])
 p=np.mean(ps,0);rows.append({'patient_id':r.patient_id,'exam_date':r.exam_date,'primary_label':int(r.primary_label),'p_normal':p[0],'p_early':p[1],'p_intermediate':p[2],'p_advanced':p[3],'pred_stage':int(p.argmax()),'pred_any_glaucoma':int(p.argmax()!=0)})
out=pd.DataFrame(rows);out.to_csv(O/'bangladesh_frozen_gleam_model_predictions.csv',index=False);pos=out[out.primary_label==1];neg=out[out.primary_label==0];res={'n_evaluable':len(out),'n_positive':len(pos),'n_negative':len(neg),'positive_sensitivity_argmax_any_glaucoma':float(pos.pred_any_glaucoma.mean()) if len(pos) else None,'negative_specificity_argmax':float((neg.pred_any_glaucoma==0).mean()) if len(neg) else None,'predicted_stage_counts':out.pred_stage.value_counts().sort_index().to_dict(),'interpretation':'Direct frozen GLEAM ensemble domain-shift sensitivity analysis; incompatible visit-level binary endpoint and one negative preclude external validation.'};(O/'bangladesh_frozen_gleam_model_sensitivity.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
