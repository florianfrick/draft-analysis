"""`model` is a regular package (this __init__ makes it one).

Why this file exists
--------------------
This repo has two modules literally named ``model.py``
(``model/finetune/model.py`` and ``model/policy/model.py``). When a script in
one of those directories is run directly, Python puts that directory on
``sys.path[0]``; the local ``model.py`` then *shadows* the ``model/`` package
and ``import model`` / ``from model.pretraining ...`` break. Making ``model`` a
real package (with this __init__) means ``import model`` always resolves to this
directory, and the two ``model.py`` files are reachable only as submodules
(``model.finetune.model``, ``model.policy.model``).

The three public model classes are re-exported here so the legacy shortcuts
``from model import ProDraftModel`` / ``PolicyValueModel`` / ``PrefixValueModel``
keep working regardless of which directory a caller lives in. Every entry
script also inserts the *repo root* (the parent of this ``model/`` directory) -
not ``model/`` itself - onto ``sys.path`` so the package is found.
"""

from .finetune.model import PrefixValueModel, ProDraftModel
from .policy.model import PolicyValueModel

__all__ = ["ProDraftModel", "PrefixValueModel", "PolicyValueModel"]
