from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
O=Path('output/gleam_fourclass');F=Path('submission_medical_image_analysis/figures');F.mkdir(parents=True,exist_ok=True)
perf=pd.read_csv(O/'ensemble_performance.csv');cost=pd.read_csv(O/'computational_cost.csv');abl=pd.read_csv(O/'component_ablations_exploratory.csv')
names={'early_fusion':'Early fusion','late_fusion':'Late fusion','gated_token':'Proposed','official_hamm':'HAMM reproduction'}

fig,ax=plt.subplots(figsize=(7.2,4.3));x=np.arange(len(perf));bars=ax.bar(x,perf.macro_auc,color=['#8d99ae','#6c757d','#0068b5','#d98e04']);ax.set_ylim(.86,.95);ax.set_ylabel('Macro one-vs-rest ROC AUC');ax.set_xticks(x,[names[m] for m in perf.model],rotation=12);ax.bar_label(bars,fmt='%.3f',padding=3);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(F/'model_performance.png',dpi=300);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.8,4.4));x=cost.parameters/1e6;y=perf.set_index('model').loc[cost.model].macro_auc;sz=cost.peak_gpu_mb_mean/8
for i,r in cost.iterrows():ax.scatter(x[i],y.iloc[i],s=sz.iloc[i],alpha=.72);ax.annotate(names[r.model],(x[i],y.iloc[i]),xytext=(5,5),textcoords='offset points',fontsize=9)
ax.set_xlabel('Trainable parameters (millions)');ax.set_ylabel('Macro ROC AUC');ax.set_ylim(.89,.945);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(F/'efficiency_tradeoff.png',dpi=300);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.2));full=json.loads((O/'gated_token_seed29_result.json').read_text())['test']['auc_macro_ovr'];vals=[full]+abl.auc_macro_ovr.tolist();labs=['Complete','No gates','No Transformer','No regularizers'];bars=ax.bar(labs,vals,color=['#0068b5','#88bde6','#88bde6','#88bde6']);ax.set_ylim(.88,.95);ax.set_ylabel('Macro ROC AUC (seed 29)');ax.bar_label(bars,fmt='%.3f',padding=3);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(F/'component_ablation.png',dpi=300);plt.close(fig)

# Architecture and graphical abstract
fig,ax=plt.subplots(figsize=(12,4));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,color):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#334155',linewidth=1.2);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,weight='bold')
for y,t,c in [(2.9,'SLO','#dbeafe'),(1.75,'OCT RNFL thickness','#dcfce7'),(.6,'Visual-field PD map','#fef3c7')]:box(.2,y,1.8,.65,t,c);box(2.5,y,1.5,.65,'ResNet-18',c);box(4.5,y,1.5,.65,'Projection\n+ gate',c);ax.add_patch(FancyArrowPatch((2,y+.33),(2.5,y+.33),arrowstyle='->',mutation_scale=12));ax.add_patch(FancyArrowPatch((4,y+.33),(4.5,y+.33),arrowstyle='->',mutation_scale=12))
box(6.7,1.45,1.8,1.1,'Modality tokens\n+ embeddings','#ede9fe');box(9.1,1.45,1.4,1.1,'Transformer\nfusion','#fce7f3');box(11,1.45,.8,1.1,'4-stage\noutput','#fee2e2')
for y in [3.23,2.08,.93]:ax.add_patch(FancyArrowPatch((6,y),(6.7,2),arrowstyle='->',mutation_scale=12));ax.add_patch(FancyArrowPatch((8.5,2),(9.1,2),arrowstyle='->',mutation_scale=12));ax.add_patch(FancyArrowPatch((10.5,2),(11,2),arrowstyle='->',mutation_scale=12));ax.text(7.6,.35,'Auxiliary supervision and modality dropout during training',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(F/'architecture.png',dpi=300,bbox_inches='tight');fig.savefig(F/'graphical_abstract.png',dpi=300,bbox_inches='tight');plt.close(fig)

# Copy analysis figures.
import shutil
for n in ['calibration_decision_curve.png','confusion_matrices.png']:shutil.copy2(O/n,F/n)
print('figures_created',len(list(F.glob('*.png'))))
