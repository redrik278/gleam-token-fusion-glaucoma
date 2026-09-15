from pathlib import Path
import json,pandas as pd,matplotlib.pyplot as plt,numpy as np
O=Path('output/gleam_fourclass');F=Path('submission_medical_image_analysis/figures');F.mkdir(exist_ok=True)
d=pd.read_csv(O/'reviewer_baseline_ensemble_summary.csv').set_index('model')
fig,ax=plt.subplots(1,2,figsize=(10.8,3.6))
mods=['vf','vf_oct','vf_oct_slo'];labs=['VF','VF+OCT','VF+OCT+SLO'];v=d.loc[mods]
ax[0].plot(labs,v.auc,'o-',lw=2,label='Macro AUC');ax[0].plot(labs,v.accuracy,'s--',lw=2,label='Accuracy');ax[0].set_ylim(.64,.95);ax[0].set_title('(a) Independently trained modality addition');ax[0].legend(frameon=False);ax[0].grid(axis='y',alpha=.25)
mods=['weighted_logits','attention_pool','mlp_matched','late_fusion','gated_token','token_no_gates'];labs=['Weighted\nlogits','Attention\npooling','MLP','Late\nfusion','Gated\ntokens','Ungated\ntokens'];v=d.loc[mods]
x=np.arange(len(mods));ax[1].bar(x,v.auc,color=['#94a3b8','#94a3b8','#60a5fa','#60a5fa','#f97316','#fb923c']);ax[1].set_xticks(x,labs);ax[1].set_ylim(.90,.945);ax[1].set_ylabel('Macro ROC AUC');ax[1].set_title('(b) Simple and token fusion');ax[1].grid(axis='y',alpha=.25)
for i,z in enumerate(v.auc):ax[1].text(i,z+.0008,f'{z:.3f}',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(F/'reviewer_baselines.png',dpi=350,bbox_inches='tight')
