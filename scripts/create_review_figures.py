from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt

O=Path('output/gleam_fourclass');F=Path('submission_medical_image_analysis/figures');F.mkdir(parents=True,exist_ok=True)
clinical=json.loads((O/'clinical_metrics_extended.json').read_text());missing=pd.read_csv(O/'missing_modality_performance.csv');pert=pd.read_csv(O/'anatomical_perturbation_results.csv');additional=pd.read_csv(O/'additional_model_summary.csv')
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8})
fig,axs=plt.subplots(2,2,figsize=(11.5,7.7));stages=['Normal','Early','Intermediate','Advanced'];colors=['#4575b4','#91bfdb','#fdae61','#d73027']
# Stage metrics with bootstrap CIs.
ax=axs[0,0];x=np.arange(4);width=.25
for j,(key,label) in enumerate([('sensitivity','Sensitivity'),('f1','F1'),('auprc','AUPRC')]):
 vals=np.array([clinical['stage_metrics'][str(i)][key] for i in range(4)]);cis=np.array([clinical['stage_metrics'][str(i)][key+'_ci95'] for i in range(4)]);err=np.vstack([vals-cis[:,0],cis[:,1]-vals]);ax.bar(x+(j-1)*width,vals,width,label=label,yerr=err,capsize=2)
ax.set_xticks(x,stages);ax.set_ylim(0,1.08);ax.set_ylabel('Score (95% eye-sample bootstrap CI)');ax.set_title('(a) Stage-specific performance');ax.legend(ncol=3,frameon=False)
# Missing modality.
ax=axs[0,1];m=missing.iloc[::-1];ax.barh(m.available_modalities,m.macro_auc,color='#4c78a8');ax.set_xlim(.74,.95);ax.set_xlabel('Macro one-vs-rest AUC');ax.set_title('(b) Missing-modality inference');ax.grid(axis='x',alpha=.2)
# Augmentation and ordinal comparison.
ax=axs[1,0];names={'gated_token':'Primary gated token','gated_token_clinical_aug':'Modality-specific augmentation','ordinal_token':'Ordinal cumulative link'};x=np.arange(3);w=.24
for j,(key,label) in enumerate([('macro_auc','Macro AUC'),('accuracy','Accuracy'),('macro_f1','Macro F1')]):ax.bar(x+(j-1)*w,additional[key],w,label=label)
ax.set_xticks(x,[names[v] for v in additional.model],rotation=8);ax.set_ylim(.35,1);ax.set_title('(c) Objective and augmentation analyses');ax.legend(ncol=3,frameon=False)
# Perturbations.
ax=axs[1,1];pivot=pert.pivot(index='modality',columns='occluded_region',values='auc_change_from_unperturbed').loc[['SLO','OCT','VF']];x=np.arange(3)
ax.bar(x-.18,-pivot['clinical_roi'],.36,label='Clinical-region occlusion',color='#d95f02');ax.bar(x+.18,-pivot['border'],.36,label='Border occlusion',color='#7570b3');ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['SLO','OCT','Visual field']);ax.set_ylabel('Decrease in macro AUC');ax.set_title('(d) Anatomical/shortcut perturbation');ax.legend(frameon=False)
fig.tight_layout();fig.savefig(F/'review_evidence.png',dpi=400,bbox_inches='tight');plt.close(fig)

# Copy XAI examples generated from the frozen checkpoint.
import shutil
shutil.copy2(O/'gradcam_examples.png',F/'gradcam_examples.png')
print('created',F/'review_evidence.png',F/'gradcam_examples.png')
