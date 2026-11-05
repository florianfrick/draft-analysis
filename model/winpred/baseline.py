"""The baseline: one number, the blue-side win rate of the training split.

`p = mean(blue_win | train)`, the same value for every game. Nothing else - no
champion, no patch, no side interaction. That is the point: it is the null a
draft model has to beat, and it is *strong* on this corpus because the side
advantage is real (pro 53.05% blue, soloQ 47.64% blue on the current data).

Two consequences worth stating, because they are what make the comparison
honest rather than flattering:

* **AUC is exactly 0.5.** A constant predictor has every score tied, so it
  ranks nothing. Any AUC above 0.5 is therefore entirely the draft model's, and
  that is the cleanest single statement the experiment can make.
* **Accuracy is `max(p, 1-p)`**, i.e. "always call the more common side". On
  pro that is 53.05% (call blue), on soloQ 52.36% (call *red*, because blue
  wins less than half at high elo). Those are the numbers the brief expects
  ("about 53% correct") and they are not the same side - which is itself a
  result, and the reason the prior must be fitted per dataset rather than
  assumed to be 0.5.

`side_bias` in the model is this same degree of freedom, learned instead of
counted, which is why the baseline is the right comparison: it isolates exactly
"what does the draft add on top of the side prior".
"""
import json
from pathlib import Path

import numpy as np


class BlueWinRateBaseline:
    """Constant P(blue wins) = the training split's blue win rate."""

    def __init__(self, prior=None, n_train=0):
        self.prior = None if prior is None else float(prior)
        self.n_train = int(n_train)

    def fit(self, y_train):
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if y.size == 0:
            raise ValueError("cannot fit a prior on zero rows")
        bad = np.unique(y[~np.isin(y, (0.0, 1.0))])
        if bad.size:
            raise ValueError("labels must be 0/1, found %s" % bad[:5])
        self.prior = float(y.mean())
        self.n_train = int(y.size)
        return self

    def predict_proba(self, n):
        if self.prior is None:
            raise RuntimeError("baseline is not fitted")
        return np.full(int(n), self.prior, dtype=np.float64)

    def describe(self):
        """JSON-able description, written into every report that uses it."""
        if self.prior is None:
            raise RuntimeError("baseline is not fitted")
        return {"name": "blue_win_rate_prior",
                "prior": self.prior,
                "n_train": self.n_train,
                "predicted_side": "blue" if self.prior > 0.5 else "red",
                "expected_accuracy": max(self.prior, 1.0 - self.prior),
                "note": "one scalar, fitted on the training split only; "
                        "AUC is 0.5 by construction"}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.describe(), f, indent=1)
            f.write("\n")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(prior=payload["prior"], n_train=payload.get("n_train", 0))


def fit_prior(table, train_idx):
    """Convenience: fit on a table's train rows."""
    return BlueWinRateBaseline().fit(table.y[np.asarray(train_idx)])
