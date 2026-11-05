"""Win prediction from a completed draft (branch exp/win-prediction).

Self-contained on purpose: everything this experiment needs lives in this
folder, so the whole thing can be moved into `main` (or dropped) without
touching the existing stages. The only things it imports from the repo are the
*data pipelines* (`data/pro play/pipeline.py`, `data/solo queue/.../pipeline.py`)
and the *encoder blocks* (`model/pretraining/soloq_model.py`) - both read-only.

Making this a package matters for `pytest model/winpred`: without
`__init__.py` pytest puts this directory on `sys.path`, and `model.py` here
would then shadow the `model/` package (the trap documented in
`model/__init__.py`).

Files
-----
data.py       pro + soloQ full-draft tables, the 80/10/10 split, prediction I/O
metrics.py    log-loss, Brier, ECE, ROC-AUC, precision/recall at high confidence
baseline.py   the blue-side win-rate prior
model.py      the Siamese draft encoder + outcome MLP
train.py      training (CE or focal), W&B, checkpoint + test predictions
evaluate.py   score/compare any mix of checkpoints, prediction files, baseline
"""
