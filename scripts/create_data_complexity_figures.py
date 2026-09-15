from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from PIL import Image
O=Path('output/gleam_fourclass');D=Path('data/public/GLEAM/split/testset');F=Path('submission_medical_image_analysis/figures');F.mkdir(exist_ok=True)
labels=['Normal','Early','Intermediate','Advanced']; seeds=[17,29,43]
dfs=[pd.read_csv(O/f'gated_token_seed{s}_test.csv').sort_values('id') for s in seeds]
d=dfs[0][['id','label']].copy(); ps=np.mean([x[[f'p{i}' for i in range(4)]].to_numpy() for x in dfs],0);d['pred']=ps.argmax(1);d['conf']=ps.max(1)
sel=[]
for c in range(4):
 q=d[(d.label==c)&(d.pred==c)].sort_values('conf',ascending=False);sel.append(q.iloc[len(q)//2])
fig,ax=plt.subplots(4,4,figsize=(9.2,8.3),gridspec_kw={'width_ratios':[1,1,1,1.25]})
for r,row in enumerate(sel):
 sid=f'{int(row.id):04d}'
 for j,(fn,title) in enumerate(zip(['SLO.jpg','Thickness.jpg','VF.jpg'],['SLO','OCT','VF'])):
  ax[r,j].imshow(Image.open(D/sid/fn).convert('RGB'));ax[r,j].axis('off');
  if r==0:ax[r,j].set_title(title,fontsize=10,weight='bold')
 p=ps[d.index[d.id==row.id][0]];colors=['#2563eb' if i==int(row.label) else '#cbd5e1' for i in range(4)]
 ax[r,3].barh(np.arange(4),p,color=colors);ax[r,3].set_xlim(0,1);ax[r,3].set_yticks(range(4),['N','E','I','A']);ax[r,3].invert_yaxis();ax[r,3].grid(axis='x',alpha=.2);ax[r,3].set_xlabel('Probability',fontsize=8)
 if r==0:ax[r,3].set_title('Ensemble output',fontsize=10,weight='bold')
 ax[r,0].text(-.08,.5,f'{labels[r]}\nID {sid}',transform=ax[r,0].transAxes,ha='right',va='center',fontsize=9,weight='bold')
fig.suptitle('Representative held-out GLEAM inputs and model outputs',fontsize=12,weight='bold');fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(F/'data_prediction_examples.png',dpi=350,bbox_inches='tight');plt.close(fig)

cost=pd.read_csv(O/'computational_cost.csv').set_index('model');perf=pd.read_csv(O/'ensemble_performance.csv').set_index('model');models=['early_fusion','late_fusion','gated_token','official_hamm'];names=['Early fusion','Late fusion','Gated tokens','HAMM reproduction'];cols=['#3b82f6','#10b981','#f97316','#8b5cf6']
# Higher is better on every spoke. Efficiency is min(observed burden)/model burden.
vals=[]
for m in models: vals.append([perf.loc[m,'macro_auc'],perf.loc[m,'accuracy'],cost.parameters.min()/cost.loc[m,'parameters'],cost.peak_gpu_mb_mean.min()/cost.loc[m,'peak_gpu_mb_mean'],cost.training_seconds_mean.min()/cost.loc[m,'training_seconds_mean']])
vals=np.array(vals);axes=['Macro AUC','Accuracy','Parameter\nefficiency','Memory\nefficiency','Runtime\nefficiency'];ang=np.linspace(0,2*np.pi,len(axes),endpoint=False);ang=np.r_[ang,ang[0]]
fig=plt.figure(figsize=(6.4,5.7));a=fig.add_subplot(111,polar=True)
for v,n,c in zip(vals,names,cols):vv=np.r_[v,v[0]];a.plot(ang,vv,lw=2,label=n,color=c);a.fill(ang,vv,alpha=.06,color=c)
a.set_xticks(ang[:-1],axes,fontsize=9);a.set_ylim(0,1);a.set_yticks([.2,.4,.6,.8,1]);a.set_yticklabels(['.2','.4','.6','.8','1.0'],fontsize=7);a.set_title('Performance–complexity profile\n(higher is better)',pad=18,fontsize=12,weight='bold');a.legend(loc='lower center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,fontsize=8);fig.tight_layout();fig.savefig(F/'model_complexity_radar.png',dpi=350,bbox_inches='tight');plt.close(fig)
pd.DataFrame(vals,index=names,columns=['macro_auc','accuracy','parameter_efficiency','memory_efficiency','runtime_efficiency']).to_csv(O/'model_complexity_radar_values.csv')
