#!/usr/bin/env python3
"""Export the shipped draft heads to ExecuTorch `.pte` files + static assets.

Run from the repo root, inside the project's dev container:

    docker compose run --rm -T dev bash -c \
      'PYTHONPATH=/root/.cache/ettools python android/tools/export_offline.py'

The script writes into `android/app/src/main/assets/`:

    policy.pte          naive transformer: 173 next-move logits + 10x173 slot logits
    masker.pte          masking encoder: 10x173 role-conditioned completion logits
    offline.json        champions, roles, patches, move_plan, lane table, model labels
    icons/<riot>.png    champion portraits, copied from the dDragon cache when available

Why the graphs take explicit tensors instead of the Python dict the server uses:
`torch.export` cannot capture a dict with string keys plus the Python-side
branching in `build_board`. Board construction is deterministic and already
ported to Kotlin (`DraftOrder`, plus the new `LocalEngine`), so the boundary is
moved there and the graph starts at fixed-shape tensors.

The logits are deliberately NOT masked here. Both heads apply their own
legality rules to logits inside `forward`; keeping that in Kotlin means one
implementation of "-1e4 before top-k" serves both local and remote paths, and
the role head can reuse one forward for all ten cells.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

# ExecuTorch is installed into the dev container's persistent cache, not the
# project image. Setting PYTHONPATH on the Git Bash command line rewrites
# `/root/...` into a Windows path, so set it here where the path is unambiguous.
for tool_path in ("/root/.cache/ettools",):
    if tool_path not in sys.path:
        sys.path.insert(0, tool_path)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: E402
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # noqa: E402
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.utils.configs import get_transform_passes  # noqa: E402
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from torch.export import export as torch_export  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

from pro.model.naivetransformer import model as nt  # noqa: E402
from solo_queue.model.policy import infer  # noqa: E402
from solo_queue.model.policy import lanes  # noqa: E402
from solo_queue.model.pretraining.base_model import (  # noqa: E402
    SiameseBase,
    attr_matrix_for_config,
    load_encoder_state,
)

ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"
POLICY_CKPT = ROOT / ("pro/model/checkpoints/"
                      "nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw/best.pt")
MASKER_CKPT = ROOT / "pro/model/checkpoints/masking-proonly-nov13-through-lcp-df-sw/best.pt"
DDRAGON = "https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/"
MANIFEST_CACHE = ROOT / "webapp" / ".cache" / "champion_16.20.1.json"
PROC = ROOT / "data" / "solo queue" / "process matchdata"

N_SLOTS = 10
FEARLESS_SLOTS = 40


# --------------------------------------------------------------------- graphs
class PolicyGraph(nn.Module):
    """Naive transformer -> (policy logits [1,173], slot logits [1,10,173]).

    Fixed shapes are intentional: one graph, no re-tracing per board. Kotlin
    supplies all fifteen tensors from the same board the server would see.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        champ_ids: torch.Tensor,
        ban_ids: torch.Tensor,
        fearless_ids: torch.Tensor,
        fearless_role_ids: torch.Tensor,
        role_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        move_idx: torch.Tensor,
        phase: torch.Tensor,
        to_move: torch.Tensor,
        first_pick: torch.Tensor,
        game_num: torch.Tensor,
        is_ban: torch.Tensor,
        patch_id: torch.Tensor,
    ):
        batch = {
            "champ_ids": champ_ids,
            "ban_ids": ban_ids,
            "fearless_ids": fearless_ids,
            "fearless_role_ids": fearless_role_ids,
            "role_ids": role_ids,
            # Android's ExecuTorch Tensor API has no fromBlob(BooleanArray):
            # bool masks arrive as int64 0/1 and are cast here. Keeping the
            # cast INSIDE the graph means Kotlin feeds every input as LongArray
            # and the exported method signature never changes with dtype
            # quirks in the runtime.
            "pad_mask": pad_mask.bool(),
            "move_idx": move_idx,
            "phase": phase,
            "to_move": to_move,
            "first_pick": first_pick,
            "game_num": game_num,
            "is_ban": is_ban,
            "patch_id": patch_id,
        }
        hidden = self.model.encode(batch)
        normed = self.model.head_norm(hidden)
        return (
            self.model.policy_head(normed[:, nt.STATE_POS, :]),
            self.model.policy_head(normed[:, nt.PICK_START:nt.PICK_START + nt.N_PICKS, :]),
        )


class MaskerGraph(nn.Module):
    """Masking encoder -> masked-slot champion logits [1,10,173]."""

    def __init__(self, model: nn.Module, num_champs: int):
        super().__init__()
        self.model = model
        self.num_champs = int(num_champs)

    def forward(
        self,
        champ_ids: torch.Tensor,
        role_ids: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        batch = {
            "champ_ids": champ_ids,
            "role_ids": role_ids,
            "is_masked": champ_ids == self.num_champs,
            "pad_mask": pad_mask.bool(),
        }
        return self.model(batch, compute_heads=True, compute_value=False)["champ_logits"]


# -------------------------------------------------------------------- fixtures
def policy_inputs(num_champs: int, n_patches: int):
    pad = num_champs + 1
    long = lambda n, fill: torch.full((1, n), fill, dtype=torch.long)
    scalar = lambda fill: torch.full((1,), fill, dtype=torch.long)
    return (
        long(N_SLOTS, pad),                    # champ_ids
        long(N_SLOTS, pad),                    # ban_ids
        long(FEARLESS_SLOTS, pad),             # fearless_ids
        long(FEARLESS_SLOTS, nt.ROLE_FLEX_ID),  # fearless_role_ids
        long(N_SLOTS, nt.ROLE_FLEX_ID),        # role_ids
        long(N_SLOTS, 1),                      # pad_mask (int64 0/1 -> .bool())
        scalar(0), scalar(0), scalar(0), scalar(0),
        scalar(1), scalar(0), scalar(1 if n_patches else 0),
    )


def masker_inputs(num_champs: int):
    pad = num_champs + 1
    ids = torch.full((1, N_SLOTS), pad, dtype=torch.long)
    ids[0, 0] = num_champs                      # the slot being completed
    roles = torch.full((1, N_SLOTS), nt.ROLE_FLEX_ID, dtype=torch.long)
    roles[0, 0] = 0                             # TOP
    return (ids, roles, torch.zeros(1, N_SLOTS, dtype=torch.long))


# --------------------------------------------------------------------- export
def export_pte(module: nn.Module, sample, out: Path, quantized: bool = True):
    torch.manual_seed(0)
    print(f"exporting {out.name} (quantized={quantized})", flush=True)
    with torch.no_grad():
        traced = torch_export(module.eval(), sample, strict=False)

    if quantized:
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True))
        prepared = prepare_pt2e(traced.module(), quantizer)
        # Dynamic quantization needs no calibration; still run one sample so
        # observers see the real shapes before convert.
        with torch.no_grad():
            prepared(*sample)
        converted = convert_pt2e(prepared)
        traced = torch_export(converted, sample, strict=False)

    program = to_edge_transform_and_lower(
        traced,
        partitioner=[XnnpackPartitioner()],
        transform_passes=get_transform_passes(),
    ).to_executorch()
    with open(out, "wb") as handle:
        program.write_to_file(handle)
    print(f"  wrote {out.stat().st_size / 1e6:.1f} MB", flush=True)


# ------------------------------------------------------------------- metadata
def static_metadata(policy_blob, masker_blob):
    mapping = json.loads((PROC / "champ_mapping.json").read_text())["riot_to_idx"]
    names = json.loads((PROC / "champ_names.json").read_text())["names"]
    riot_to_idx = {int(k): int(v) for k, v in mapping.items()}
    riot_to_name = {int(k): v for k, v in names.items()}

    manifest = None
    if MANIFEST_CACHE.is_file():
        manifest = json.loads(MANIFEST_CACHE.read_text()).get("data") or {}
    champions = []
    for riot, name in sorted(riot_to_name.items(), key=lambda kv: kv[1].lower()):
        champions.append({
            "idx": riot_to_idx[riot],
            "riot": riot,
            "name": name,
            "img": "icons/%d.png" % riot,
        })

    vocab = [str(k) for k in (policy_blob.get("patch_vocab") or [])]
    patches = [{"id": i + 1, "key": k} for i, k in enumerate(vocab)][::-1]
    default = next((p["id"] for p in patches if p["key"] == infer.DEFAULT_PATCH),
                   patches[0]["id"] if patches else 0)

    plans = {
        str(fp): [["ban" if kind == infer.MOVE_BAN else "pick", int(team)]
                  for kind, team in infer.move_plan(fp)]
        for fp in (0, 1)
    }

    t = lanes.table()
    return {
        "champions": champions,
        "roles": list(t["roles"]),
        "move_plan": plans,
        "patches": patches,
        "default_patch_id": default,
        "default_patch": infer.DEFAULT_PATCH,
        "lane_counts": {str(k): v for k, v in t["counts"].items()},
        "lane_share": t["share"],
        "lane_floor": t["floor"],
        "models": {
            "policy": {
                "ckpt": str(POLICY_CKPT.relative_to(ROOT)),
                "arch": "naive_transformer",
                "val_top5": (policy_blob.get("metrics") or {}).get("top5"),
            },
            "masker": {
                "ckpt": str(MASKER_CKPT.relative_to(ROOT)),
                "arch": "masking",
            },
        },
    }


def copy_icons(champions):
    out = ASSETS / "icons"
    out.mkdir(parents=True, exist_ok=True)
    cached = set()
    if MANIFEST_CACHE.is_file():
        data = json.loads(MANIFEST_CACHE.read_text()).get("data") or {}
        for row in data.values():
            try:
                cached.add((int(row["key"]), row["image"]["full"]))
            except (KeyError, TypeError, ValueError):
                pass
    by_riot = dict(cached)
    copied = 0
    for champ in champions:
        riot = champ["riot"]
        target = out / ("%d.png" % riot)
        if target.is_file():
            continue
        filename = by_riot.get(riot) or ("%s.png" % champ["name"])
        try:
            with urllib.request.urlopen(DDRAGON + filename, timeout=10) as response:
                target.write_bytes(response.read())
            copied += 1
        except Exception as exc:  # noqa: BLE001 - offline export still usable
            print("  icon %s: %s" % (filename, exc))
    print("  icons copied: %d (total %d)" % (copied, len(list(out.glob("*.png")))))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    policy_blob = torch.load(POLICY_CKPT, map_location="cpu", weights_only=False)
    masker_blob = torch.load(MASKER_CKPT, map_location="cpu", weights_only=False)

    policy_cfg = policy_blob["config"]
    policy_model = nt.build_from_config(policy_cfg)
    policy_model.load_state_dict(policy_blob["model"], strict=True)
    policy = PolicyGraph(policy_model)
    policy_champs = int(policy_cfg["model"]["num_champs"])
    n_patches = int(policy_cfg.get("n_patches") or 0)

    masker_cfg = masker_blob["config"]
    state = masker_blob["model"]
    if any(k.startswith("encoder.") for k in state):
        state = {k[len("encoder."):]: v for k, v in state.items()
                 if k.startswith("encoder.")}
    masker_model = SiameseBase(
        num_champs=int(masker_cfg["num_champs"]),
        embed_dim=int(masker_cfg["embed_dim"]),
        num_heads=int(masker_cfg["num_heads"]),
        num_intra_layers=int(masker_cfg["num_intra"]),
        num_cross_layers=int(masker_cfg["num_cross"]),
        n_patches=int(masker_cfg.get("n_patches", 0)),
        attr_matrix=attr_matrix_for_config(masker_cfg),
    )
    load_encoder_state(masker_model, state)
    masker = MaskerGraph(masker_model, masker_cfg["num_champs"])

    n_patches = int(policy_cfg.get("n_patches") or 0)
    with torch.no_grad():
        policy_sample = policy_inputs(policy_champs, n_patches)
        masker_sample = masker_inputs(masker_cfg["num_champs"])
        p0, s0 = policy(*policy_sample)
        m0 = masker(*masker_sample)
        print("policy logits:", tuple(p0.shape), tuple(s0.shape),
              "top1:", int(p0[0].argmax()))
        print("masker logits:", tuple(m0.shape),
              "top1 slot0:", int(m0[0, 0].argmax()))

    export_pte(policy, policy_sample, ASSETS / "policy.pte")
    # The masker's quantized graph fails ExecuTorch 1.5's out-variant pass
    # (quantized_decomposed choose_qparams/quantize/dequantize). Its fp32
    # weights are ~10 MB, so shipping fp32 is the honest, smaller-risk choice.
    export_pte(masker, masker_sample, ASSETS / "masker.pte", quantized=False)

    metadata = static_metadata(policy_blob, masker_blob)
    (ASSETS / "offline.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_icons(metadata["champions"])
    print("wrote", ASSETS / "offline.json")


if __name__ == "__main__":
    main()
