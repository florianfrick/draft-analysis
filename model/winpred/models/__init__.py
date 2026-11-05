"""The models this experiment compares.

`sweep.py` and `evaluate.py` treat a checkpoint as a black box, so the point of
this package is that adding an architecture is one file and no plumbing.

siamese_model.py   the draft encoder -> P(blue wins) (the model under test)
baseline.py        the blue-side win-rate prior (the null it has to beat)

Both are re-exported here so callers can `from model.winpred.models import
WinPredictor, BlueWinRateBaseline`; `siamese_model` stays importable as a
module because the trainers use its checkpoint helpers (`config_of`,
`build_from_config`, `load_checkpoint`, `param_count`).
"""
from model.winpred.models.baseline import BlueWinRateBaseline, fit_prior
from model.winpred.models.siamese_model import (RoleAwareCrossAttention,
                                                WinPredictor, build_from_config,
                                                config_of, load_checkpoint,
                                                param_count)

__all__ = ["BlueWinRateBaseline", "fit_prior", "WinPredictor",
           "RoleAwareCrossAttention", "build_from_config", "config_of",
           "load_checkpoint", "param_count"]
