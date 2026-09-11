"""Three-seed ensemble analysis and paired eye-sample inference for ablations."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, log_loss, roc_auc_score,
)

OUT = Path("output/gleam_fourclass")
SEEDS = (17, 29, 43)
MODELS = ("gated_token", "token_no_gates", "token_no_transformer", "token_no_regularizers")
LABELS = {
    "gated_token": "Complete model",
    "token_no_gates": "No gates",
    "token_no_transformer": "No Transformer",
    "token_no_regularizers": "No aux./dropout",
}


def load_ensemble(model):
    frames = [pd.read_csv(OUT / f"{model}_seed{s}_test.csv", dtype={"id": str}) for s in SEEDS]
    base = frames[0]
    for frame in frames[1:]:
        if not np.array_equal(base["id"].to_numpy(), frame["id"].to_numpy()):
            raise RuntimeError(f"Sample order mismatch for {model}")
        if not np.array_equal(base.label.to_numpy(), frame.label.to_numpy()):
            raise RuntimeError(f"Label mismatch for {model}")
    return base.label.to_numpy(), np.mean([
        f[[f"p{i}" for i in range(4)]].to_numpy() for f in frames
    ], axis=0)


def metrics(y, p):
    pred = p.argmax(1)
    onehot = np.eye(4)[y]
    confidence = p.max(1)
    correct = pred == y
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lo) & (confidence < (hi if hi < 1 else hi + 1e-12))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "macro_auc": roc_auc_score(y, p, multi_class="ovr", average="macro"),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "nll": log_loss(y, p, labels=np.arange(4)),
        "brier": np.mean(np.sum((p - onehot) ** 2, axis=1)),
        "ece": ece,
    }


ensembles = {}
y = None
for model in MODELS:
    yy, ensembles[model] = load_ensemble(model)
    if y is None:
        y = yy
    elif not np.array_equal(y, yy):
        raise RuntimeError("Labels differ across variants")

point = {model: metrics(y, p) for model, p in ensembles.items()}
rng = np.random.default_rng(20260911)
B = 10000
auc_boot = {model: [] for model in MODELS}
delta_boot = {model: [] for model in MODELS[1:]}
for _ in range(B):
    ix = rng.integers(0, len(y), len(y))
    yy = y[ix]
    if len(np.unique(yy)) != 4:
        continue
    values = {model: roc_auc_score(yy, p[ix], multi_class="ovr", average="macro")
              for model, p in ensembles.items()}
    for model, value in values.items():
        auc_boot[model].append(value)
    for model in MODELS[1:]:
        delta_boot[model].append(values["gated_token"] - values[model])

rows = []
stats = {
    "resampling_unit": "GLEAM eye-level sample (public release has no patient-group mapping)",
    "seeds": list(SEEDS), "bootstrap_replicates": B, "variants": {}, "paired_comparisons": {}
}
for model in MODELS:
    ci = np.quantile(auc_boot[model], [.025, .975])
    row = {"variant": model, "display_name": LABELS[model], **point[model],
           "macro_auc_ci95_low": ci[0], "macro_auc_ci95_high": ci[1]}
    rows.append(row)
    stats["variants"][model] = {**point[model], "macro_auc_ci95": ci.tolist()}

full_pred = ensembles["gated_token"].argmax(1)
for model in MODELS[1:]:
    other_pred = ensembles[model].argmax(1)
    b = int(((full_pred == y) & (other_pred != y)).sum())
    c = int(((full_pred != y) & (other_pred == y)).sum())
    delta = point["gated_token"]["macro_auc"] - point[model]["macro_auc"]
    stats["paired_comparisons"][model] = {
        "auc_delta_complete_minus_ablation": delta,
        "auc_delta_ci95": np.quantile(delta_boot[model], [.025, .975]).tolist(),
        "complete_only_correct": b,
        "ablation_only_correct": c,
        "mcnemar_exact_p": float(binomtest(min(b, c), b + c, .5).pvalue) if b + c else 1.0,
    }

pd.DataFrame(rows).to_csv(OUT / "repeated_ablation_summary.csv", index=False)
(OUT / "repeated_ablation_statistics.json").write_text(json.dumps(stats, indent=2))

display = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
metric_names = [("macro_auc", "Macro AUC"), ("accuracy", "Accuracy"),
                ("macro_f1", "Macro F1"), ("qwk", "QWK")]
x = np.arange(len(display)); width = .19
for j, (column, label) in enumerate(metric_names):
    ax.bar(x + (j - 1.5) * width, display[column], width, label=label)
ax.set_xticks(x, display.display_name)
ax.set_ylim(.62, 1.0)
ax.set_ylabel("Score")
ax.set_title("Three-seed ensemble component ablations")
ax.grid(axis="y", alpha=.22)
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center")
fig.tight_layout()
fig.savefig(OUT / "ablation_multimetric_repeated.png", dpi=300)
plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(stats["paired_comparisons"], indent=2))
