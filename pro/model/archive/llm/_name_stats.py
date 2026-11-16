"""Tokenizer-only: how the 173 champion names tokenize as the model emits them."""
import json
from transformers import AutoTokenizer

from pro.model.archive.llm.common import ANSWER_PREFIX, candidate_tables, champ_names

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
names = champ_names()
lens = [len(tok(ANSWER_PREFIX + n, add_special_tokens=False)["input_ids"])
        for n in names]
hist = {}
for n in lens:
    hist[n] = hist.get(n, 0) + 1
first, owner = candidate_tables(tok, names)
sizes = owner.sum(dim=1)
print(json.dumps({
    "n_names": len(names),
    "length_histogram": dict(sorted(hist.items())),
    "single_token": sum(1 for n in lens if n == 1),
    "distinct_first_tokens": int(first.numel()),
    "collision_groups": int((sizes > 1).sum()),
    "names_in_collisions": int(sizes[sizes > 1].sum()),
    "largest_group": int(sizes.max()),
}, indent=1))
