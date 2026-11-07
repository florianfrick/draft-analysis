"""The solo-queue track: pretraining, finetune, policy, and their artefacts.

Import root is the **repo root** (the parent of this directory), so the models
are ``solo_queue.model.pretraining`` etc. This package exists so that
``import solo_queue`` cannot be shadowed by a script directory, for the same
reason ``solo_queue/model/__init__.py`` exists for ``model``: several folders
contain a module literally named ``model.py``.

The encoder defined in ``solo_queue/model/pretraining/base_model.py`` is *shared* with the
pro track: ``pro.model.proonly`` builds the same architecture from scratch on
pro drafts. Only the solo-queue *weights* are exclusive to this track.
"""
