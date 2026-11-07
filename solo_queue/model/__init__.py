"""``solo_queue.model`` - the solo-queue track's model package.

Why the package is two levels deep
----------------------------------
This repo has two modules literally named ``model.py``
(``solo_queue/model/finetune/model.py`` and ``solo_queue/model/policy/model.py``).
When a script in one of those directories is run directly, Python puts that
directory on ``sys.path[0]``, so a bare ``import model`` would resolve to the
local file and the package would vanish. Naming the package ``solo_queue.model``
fixes that by construction: no script directory is ever called ``solo_queue``,
so ``import solo_queue.model.pretraining`` cannot be shadowed. The two
``model.py`` files are reachable only as submodules
(``solo_queue.model.finetune.model``, ``solo_queue.model.policy.model``).

Every entry script inserts the **repo root** - the parent of ``solo_queue/`` -
onto ``sys.path`` so both ``solo_queue`` and ``pro`` are importable.

The three public classes are re-exported here so ``from solo_queue.model import
ProDraftModel`` works regardless of which directory a caller lives in.
"""

from .finetune.model import PrefixValueModel, ProDraftModel
from .policy.model import PolicyValueModel

__all__ = ["ProDraftModel", "PrefixValueModel", "PolicyValueModel"]
