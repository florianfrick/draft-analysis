"""The pro track: everything trained on pro drafts.

Import root is the **repo root** (the parent of this directory), so the models
are ``pro.model.proonly``, ``pro.model.archive.bans``, ``pro.model.archive.proto``, etc.

Nothing here trains on solo-queue data. ``pro.model.proonly`` imports the
*architecture* from ``solo_queue.model.pretraining.base_model`` and trains it
from scratch on pro drafts, so the cross-package import is a code dependency
only - there is no solo-queue checkpoint anywhere in the pro init path.
"""
