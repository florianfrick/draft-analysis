"""Metric accumulation for the LLM experiment.

Every number here is a ratio of two counts summed over the whole split, never
a mean of per-batch means: batches differ in how many answer tokens they carry
(1-5 per row) and in how many rows survive the legality filter, so averaging
per-batch ratios would weight a 4-token batch the same as a 2-token one.

`ft_top*` is a 136-way number and is kept in its own block so it can never be
read next to the policy head's 173-way top-5 by accident.
"""
from collections import defaultdict


class Accumulator:
    """Counts, so the split-level ratio is exact and the n is reportable."""

    def __init__(self):
        self.num = defaultdict(float)
        self.den = defaultdict(float)

    def add(self, key, num, den):
        self.num[key] += float(num)
        self.den[key] += float(den)

    def ratio(self, key):
        d = self.den[key]
        return (self.num[key] / d) if d else float("nan")

    def get(self, key):
        return self.ratio(key)

    def n(self, key):
        return self.den[key]

    def update(self, metrics, batch):
        """Fold one batch's per-example metrics in.

        `exact`, `ft_top*` are per-row rates; `ce`, `tok_acc` are per-token.
        The two are separated because a row is not a token.
        """
        B = int(batch["n_ans"].shape[0])
        n_tok = int(batch["n_ans"].sum())
        for k in ("exact", "ft_top1", "ft_top5", "ft_rank",
                  "ft_top1_masked", "ft_top5_masked", "ft_rank_masked",
                  "top1_illegal", "top5_illegal", "top1_offlist"):
            if k in metrics:
                self.add(k, metrics[k] * B, B)
        for k in ("ce", "tok_acc"):
            if k in metrics:
                self.add(k, metrics[k] * n_tok, n_tok)
        if "ft_top1_legal" in metrics:
            nl = metrics.get("legal_n", B)
            self.add("ft_top1_legal", metrics["ft_top1_legal"] * nl, nl)
            self.add("ft_top5_legal", metrics["ft_top5_legal"] * nl, nl)
            self.add("legal_n", nl, 1)
        self.add("rows", B, 1)
        self.add("tokens", n_tok, 1)

    def summary(self):
        out = {k: self.ratio(k) for k in ("ce", "tok_acc", "exact",
                                          "ft_top1", "ft_top5", "ft_rank",
                                          "ft_top1_masked", "ft_top5_masked",
                                          "ft_rank_masked",
                                          "ft_top1_legal", "ft_top5_legal",
                                          "top1_illegal", "top5_illegal",
                                          "top1_offlist")}
        # `rows` and `tokens` were added as add(key, count, 1), so the total is
        # in `num`; reading `den` reports the number of *batches* (500 for a
        # 4,000-state eval), which is the kind of off-by-a-denominator that
        # makes a report quietly wrong.
        out["n_rows"] = int(self.num.get("rows", 0))
        out["n_tokens"] = int(self.num.get("tokens", 0))
        out["n_legal_rows"] = int(self.num.get("legal_n", 0))
        return out

    def reset(self):
        self.num.clear()
        self.den.clear()


def chance_levels(n_candidates):
    """What a uniform guess scores on the first-token readout, for the report."""
    return {"top1": 1.0 / n_candidates, "top5": 5.0 / n_candidates}
