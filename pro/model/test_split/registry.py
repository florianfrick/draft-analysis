"""Which checkpoints exist, which question each answers, and whether it is comparable.

Discovery is by directory name, because the name is already the contract
(`webapp/server._ckpt_roles` reads the same prefixes):

    masking-*    masked completion / the role board      -> the "masked" suite
    nextpick-*   next move                               -> the "next_move" suite
    role-next-*  prefix + true role of the next pick     -> the "role_next" suite
    anything else                                        -> not scored here

`nextpick-naivetransformer-*` is additionally entered in `role_next`: the same
weights carry a role-conditioned slot read-out, so one checkpoint can answer
two questions without pretending the two numbers are one task.

Why a registry instead of a hand-written list: the old harness named five
checkpoints and five paths in source, so adding a model meant editing the
harness, and deleting a checkpoint left a stale path to fail at load time. Here
a new checkpoint is picked up by existing, and the only thing that needs a
decision is whether it is comparable - which is a fact about its training split,
not about the harness.

Comparability is read from the checkpoint's sibling `test_report.json`, which
the trainers write with the split they actually used. A model is comparable when
its recorded `n_train / n_val / n_test` equal the canonical split's. Anything
else is still scored - a number is better than no number - but it is labelled
`different_split` and never enters the leaderboard, because a model evaluated on
a split it trained on is not a measurement of that model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

from pro.model.test_split import split as S  # noqa: E402

# Prefix -> task. Order matters only for the error message.
PREFIX_TASK = {"masking-": "masked", "nextpick-": "next_move",
               "role-next-": "role_next"}

# Where checkpoints live. `archived` ones are scored only when asked for: they
# are the pre-`nov13` arms, trained on other splits, and their presence in a
# default run would invite exactly the cross-protocol comparison this suite
# exists to stop.
CKPT_ROOTS = [
    ("pro", ROOT / "pro" / "model" / "checkpoints", False),
    ("soloq", ROOT / "solo_queue" / "model" / "checkpoints", False),
    ("pro-archive", ROOT / "pro" / "model" / "archive" / "checkpoints", True),
]

# The incumbents. Not "the best model" - the model the frontend ships, which is
# the thing a candidate has to beat to be worth shipping. Every suite run is
# paired against these.
REFERENCE = {
    "masked": "masking-proonly-nov13-through-lcp-df-sw",
    "next_move": "nextpick-proonly-nov13-k5-through-lcp-df-sw",
    # The plain transformer is the control for the role-conditioned query:
    # the role-tuned checkpoint is the candidate being measured against it.
    "role_next": "nextpick-naivetransformer-d512-l8-h8-k5-through-lcp-df-sw",
}

# One checkpoint file per directory, in preference order. `latest.pt` is never
# preferred: same run, later step, worse on the metric the run selected on.
CKPT_FILES = ("best.pt", "best_policy.pt", "best_mask.pt")

# A non-neural baseline (`pro/model/mlp`, `pro/model/forest`) ships a `flat.json`
# instead of a torch state dict: it names the weight file, the family and the
# parameter count. Presence of this file is what makes a directory a flat
# baseline - the alternative (sniffing file extensions) would guess wrong on a
# `.pt` that is a pickled sklearn object rather than a state dict.
FLAT_MANIFEST = "flat.json"


@dataclass
class Entry:
    name: str
    path: Path
    task: str
    track: str
    archived: bool
    comparable: bool = False
    split_note: str = ""
    split_counts: dict = field(default_factory=dict)
    kind: str = "torch"
    params: int | None = None

    @property
    def is_reference(self):
        return self.name == REFERENCE.get(self.task)

    def label(self):
        """What to print in a table: the name, plus why it is not comparable."""
        if self.comparable:
            return self.name
        return "%s [%s]" % (self.name, self.split_note or "split unknown")


def task_of(name):
    for prefix, task in PREFIX_TASK.items():
        if name.startswith(prefix):
            return task
    return None


def tasks_of(name):
    """Every task one checkpoint directory is scorable on, in stable order.

    Directory prefix still decides the primary task. The transformer family is
    the one exception: its shared slot head answers the role-conditioned query
    too, so the same weights get a second `role_next` entry. `dict.fromkeys`
    keeps this idempotent if a future directory is named `role-next-*`.
    """
    tasks = []
    primary = task_of(name)
    if primary:
        tasks.append(primary)
    if name.startswith("nextpick-naivetransformer-"):
        tasks.append("role_next")
    return tuple(dict.fromkeys(tasks))


def checkpoint_file(directory):
    for fname in CKPT_FILES:
        p = directory / fname
        if p.exists():
            return p
    return None


def flat_manifest(directory):
    """`flat.json` for a baseline directory, or None.

    Separate from `checkpoint_file` because the two families answer different
    questions of a directory: a torch arm is "which of these filenames is the
    weights", a baseline is "what does its manifest say the weights are called".
    """
    path = directory / FLAT_MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _split_counts(directory):
    """n_train/n_val/n_test as the trainer recorded them, or {}."""
    report = directory / "test_report.json"
    if not report.exists():
        return {}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    proto = payload.get("protocol") or {}
    return {k: proto.get(k) for k in ("n_train", "n_val", "n_test")
            if proto.get(k) is not None}


def _reported_params(directory):
    """`protocol.params` if the trainer recorded one, else None.

    Display only. `suite.py` counts parameters off the loaded module, because
    that is the number that is true; this one is what the run *claimed*, which
    is occasionally the more interesting of the two.
    """
    report = directory / "test_report.json"
    if not report.exists():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    params = (payload.get("protocol") or {}).get("params")
    return int(params) if isinstance(params, (int, float)) else None


def _classify(entry, counts):
    want = {"n_train": S.EXPECTED["train"], "n_val": S.EXPECTED["val"],
            "n_test": S.EXPECTED["test"]}
    if not counts:
        entry.split_note = "no test_report.json"
        return
    if counts == want:
        entry.comparable = True
        return
    got = ", ".join("%s=%s" % (k.replace("n_", ""), counts.get(k, "?"))
                    for k in want)
    entry.split_note = "different split: " + got


def scan(include_archive=False, task=None):
    """Every scorable checkpoint, sorted so the reference is first."""
    out = []
    for track, root, archived in CKPT_ROOTS:
        if archived and not include_archive:
            continue
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            tasks = [t for t in tasks_of(directory.name)
                     if task is None or t == task]
            if not tasks:
                continue
            ckpt = checkpoint_file(directory)
            flat = flat_manifest(directory)
            if flat is not None:
                weights = directory / str(flat.get("weights") or "")
                if not weights.is_file():
                    continue
            elif ckpt is None:
                # An `llm-*` folder has adapter/ + projector.pt instead, and an
                # ensemble has ensemble.json; neither is a set-transformer the
                # suite can score. Skip rather than fail.
                continue
            counts = _split_counts(directory)
            for t in tasks:
                if flat is not None:
                    entry = Entry(name=directory.name, path=weights, task=t,
                                  track=track, archived=archived,
                                  kind=str(flat.get("kind") or "flat"),
                                  params=flat.get("params"))
                else:
                    entry = Entry(name=directory.name, path=ckpt, task=t,
                                  track=track, archived=archived)
                    entry.params = _reported_params(directory)
                entry.split_counts = counts
                _classify(entry, counts)
                out.append(entry)
    out.sort(key=lambda e: (not e.is_reference, e.archived, e.name))
    return out


def reference(task, entries=None):
    entries = scan(include_archive=False, task=task) if entries is None else entries
    for e in entries:
        if e.is_reference:
            return e
    return None


def missing(entries):
    return [str(e.path) for e in entries if not e.path.exists()]


def describe(entries):
    lines = []
    for e in entries:
        params = "-" if e.params is None else format(int(e.params), ",")
        lines.append("  %-56s %-10s %-6s %-8s %12s  %s"
                     % (e.name, e.task, e.track, e.kind, params,
                        "comparable" if e.comparable else e.split_note))
    return "\n".join(lines)
