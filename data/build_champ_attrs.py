"""Hand-tagged champion kit attributes -> ``data/champ_attrs.json``.

Why this exists (plan-model-v2.md §P3.1): the champion embedding table is
learned from one patch of solo-queue data, so it cannot transfer to a
champion it has never seen and cannot express anything the co-occurrence
statistics do not happen to encode. A fixed ``[173, K]`` matrix of kit
properties does both: it is a compositional prior ("engage tank", "artillery
mage"), and it gives a brand-new champion a row on patch day instead of a
random vector.

The axes are deliberately **ordinal 0..3**, not continuous. A hand-tagging
pass has maybe half a level of resolution per axis; pretending otherwise
would manufacture precision that is not there. Loaders should divide by 3
(or standardise) before concatenating into an embedding.

Anchors are written down per axis so the numbers are reproducible and
arguable. Where a champion sits between two levels it gets the middle value,
which is the intended meaning of 1 and 2 — not "low" and "high".

Source: my own knowledge of the kits, at patch 16.20. No model was queried
at build time; this file *is* the LLM tagging pass, committed so it can be
reviewed, diffed and corrected.

Revision history — each change came from review catching one column doing
two jobs:

* ``magic_damage`` -> ``magic_share`` + a new ``damage_output``. Read as a
  damage level the original was wrong: Yuumi scored 3 and deals almost
  nothing. Type and volume are independent, and drafting cares about both.
* ``peel`` -> ``peel`` (a teammate only) + a new ``kiting`` (saving
  yourself). The original conflated them, which made Kayle and Xayah look
  like peelers when what they actually do is keep dealing damage while
  being chased.
* ``sustain`` -> ``sustain`` (self only) + a new ``healing`` (allies).
  Soraka at 3 and Warwick at 3 were being called the same thing.
* Added ``side_laning`` and ``pick_potential``: holding a lane alone was
  missing entirely, and isolating one target is not engage (Blitzcrank
  dives 1 and picks 3; Malphite engages 3 and picks 1).
* ``damage_output`` capped at 1 for tanks (Cho'Gath, Dr. Mundo, Mordekaiser,
  Sion, Trundle, Volibear 2 -> 1). At 2 the axis was partly measuring
  "how much does this champion scale", which ``scaling`` already does. The
  cap is enforced in ``build()``.
* ``side_laning`` pinned as **absolute** (kit-only, not role-relative). It
  was already tagged that way; writing it down stops the next review from
  re-litigating it.

The lesson generalises: an axis that can be read two ways will be, and the
ambiguity shows up as a champion sitting next to neighbours it has nothing
in common with.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NAMES_PATH = BASE_DIR / "solo queue" / "process matchdata" / "champ_names.json"
MAPPING_PATH = BASE_DIR / "solo queue" / "process matchdata" / "champ_mapping.json"
OUT_PATH = BASE_DIR / "champ_attrs.json"

# (name, 0-anchor, 1, 2, 3-anchor). Higher is always "more of the thing".
#
# Two rules the table must satisfy, both decided by the project owner and
# both enforced in build() rather than left to the tagger's memory:
#
#   1. ``damage_output`` is a cap for tanks: any champion at tankiness 3 has
#      damage_output <= 1. A tank's job is not to be the team's damage, and
#      letting Cho'Gath/Sion/Mordekaiser sit at 2 made the axis partly a
#      "how much does this champion scale" measure instead of a role measure.
#      (Champions that deal real damage *and* soak it are tagged as bruisers,
#      tankiness 2, not as tanks.)
#   2. ``side_laning`` is **absolute, not role-relative**: the level describes
#      the kit's ability to hold a lane alone, regardless of which position
#      the champion is played in. A support played mid does not gain side
#      laning, and a top laner is not credited for the lane they happen to be
#      given.
AXES = [
    ("engage", "no way to start a fight",
     "situational or single-target initiation",
     "reliable initiation, usually on one target",
     "reliable AoE fight-start (Malphite R, Rakan W/R, Rell R)"),
    ("peel", "nothing to protect a teammate with",
     "one small tool: a slow, a single short shield",
     "solid: a displacement or a real shield on an ally",
     "elite: cancels a dive on a carry (Janna Q/R, Poppy W, Thresh lantern)"),
    ("poke", "melee, no ranged chip",
     "incidental ranged damage",
     "good lane poke / sustained ranged harassment",
     "artillery: wins the pre-fight from 900+ units"),
    ("dive", "cannot reach a backline at all",
     "needs a teammate to set it up",
     "can get in and out on its own",
     "elite backline access in a fight (Zed, Talon, Kha'Zix)"),
    ("scaling", "must win by 15 minutes or is a liability",
     "falls off noticeably after two items",
     "stable: strong at every stage",
     "late-game win condition (Kog'Maw, Vayne, Kassadin, Kayle)"),
    ("lane_priority", "concedes the wave, cannot push",
     "slow push, needs help to shove",
     "can shove and rotate on its own",
     "permapush, permanently holds the wave (Ziggs, Malzahar, Anivia)"),
    ("hard_cc", "no stun/root/knockup/suppress/charm/fear",
     "one short or conditional hard CC",
     "one reliable hard CC, or several short ones",
     "multiple reliable hard CC, chainable (Nautilus, Leona, Maokai)"),
    ("mobility", "immobile, no dash and no real self-speed",
     "one situational dash or a modest speed boost",
     "a reliable dash or blink on a short cooldown",
     "highly mobile, repeatedly repositions (Yasuo, Zed, Akali, Lee Sin)"),
    ("range", "melee",
     "short (<500): melee-with-reach, Urgot, Thresh, Rakan",
     "mid (500-650): most mages and marksmen",
     "long (650+): Caitlyn, artillery mages, Kog'Maw W, Tristana passive"),
    ("objective_speed", "cannot take a neutral objective or tower",
     "slow: a support's or tank's DPS",
     "decent sustained single-target damage",
     "elite objective DPS (Yi, Vayne, Kog'Maw, Trundle, Yorick)"),
    ("tankiness", "squishy: dies to one rotation",
     "bruiser build, survives a dive but not a teamfight",
     "real frontline: soaks a rotation and keeps going",
     "full tank: the team's wall (Ornn, Rammus, Tahm Kench)"),
    ("magic_share", "pure physical (true damage counts as neither)",
     "mostly physical, one magic spell",
     "genuinely mixed",
     "pure magic"),
    ("sustain", "no self-healing or self-shielding at all",
     "minor: a small shield or a trickle of healing",
     "meaningful: heals a real amount while trading",
     "extreme: cannot be poked or worn down (Mundo R, Garen passive)"),
    ("damage_output", "negligible: a healer or a pure enabler",
     "low: a tank's or support's contribution",
     "solid: a bruiser's or utility mage's share",
     "primary: the team's damage comes from here"),
    ("side_laning", "cannot be alone in a lane",
     "can push a wave, then must group",
     "can hold a side lane with vision",
     "dedicated split pusher who forces an answer (Fiora, Jax, Yorick)"),
    ("pick_potential", "no way to catch a lone target",
     "one short-range or unreliable catch tool",
     "a real catch tool (charm, hook, sleep, kidnap)",
     "isolates and kills one target from range (Zed, Blitzcrank, Zoe)"),
    ("kiting", "once caught, dies: no way to disengage",
     "one slow, or one short dash, not both",
     "reliably creates space while still attacking",
     "elite: keeps full damage while chased (Kalista, Vayne, Ashe, Kayle R)"),
    ("healing", "no healing or shielding for allies",
     "a token ally shield or a small heal",
     "meaningful ally healing/shielding (Karma E, Orianna E, Senna Q)",
     "elite enchanter throughput (Soraka, Yuumi, Sona, Lulu, Taric)"),
]

# name -> (engage, peel, poke, dive, scaling, lane_priority, hard_cc,
#          mobility, range, objective_speed, tankiness, magic_share,
#          sustain, damage_output, side_laning, pick_potential,
#          kiting, healing)
TABLE = {
    "Aatrox":        (1, 0, 1, 1, 1, 2, 1, 1, 1, 2, 2, 0, 2, 2, 2, 0, 1, 0),
    "Ahri":          (1, 0, 2, 2, 2, 2, 1, 3, 2, 1, 0, 3, 1, 2, 1, 3, 2, 0),
    "Akali":         (0, 0, 1, 3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 3, 2, 2, 1, 0),
    "Akshan":        (0, 0, 2, 2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 2, 2, 1, 2, 0),
    "Alistar":       (3, 3, 0, 1, 1, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 2, 0, 0),
    "Ambessa":       (1, 0, 0, 2, 2, 2, 0, 3, 0, 2, 2, 0, 1, 2, 2, 1, 1, 0),
    "Amumu":         (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 0, 2, 0, 0),
    "Anivia":        (0, 2, 2, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 2, 1, 1, 1, 0),
    "Annie":         (1, 1, 1, 1, 2, 2, 2, 0, 2, 1, 0, 3, 0, 2, 1, 2, 0, 0),
    "Aphelios":      (0, 1, 1, 0, 3, 1, 1, 0, 2, 2, 0, 0, 2, 3, 1, 0, 2, 0),
    "Ashe":          (2, 1, 2, 0, 2, 2, 2, 0, 2, 1, 0, 0, 0, 1, 1, 2, 3, 0),
    "Aurelion Sol":  (0, 1, 1, 1, 3, 2, 1, 1, 2, 1, 0, 3, 0, 3, 1, 0, 1, 0),
    "Aurora":        (1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 3, 0, 2, 1, 2, 2, 0),
    "Azir":          (1, 2, 2, 1, 3, 2, 1, 1, 2, 2, 0, 3, 0, 2, 1, 1, 2, 0),
    "Bard":          (2, 2, 1, 0, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2),
    "Bel'Veth":      (1, 0, 0, 2, 2, 2, 1, 2, 0, 3, 1, 1, 1, 3, 2, 1, 2, 0),
    "Blitzcrank":    (3, 1, 0, 1, 1, 1, 3, 1, 0, 1, 2, 2, 0, 0, 0, 3, 0, 0),
    "Brand":         (0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 0, 3, 0, 3, 1, 1, 0, 0),
    "Braum":         (1, 3, 1, 0, 2, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 1, 0, 0),
    "Briar":         (1, 0, 0, 3, 1, 2, 2, 2, 0, 2, 1, 0, 3, 2, 2, 2, 0, 0),
    "Caitlyn":       (0, 1, 3, 0, 1, 3, 1, 1, 3, 2, 0, 0, 0, 2, 1, 1, 3, 0),
    "Camille":       (1, 1, 0, 2, 2, 2, 2, 3, 0, 2, 2, 0, 1, 2, 3, 2, 2, 0),
    "Cassiopeia":    (0, 1, 2, 0, 3, 2, 1, 1, 2, 2, 0, 3, 0, 3, 1, 1, 2, 0),
    "Cho'Gath":      (1, 1, 1, 0, 3, 2, 2, 0, 0, 2, 3, 2, 2, 1, 2, 2, 0, 0),
    "Corki":         (0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 1, 1, 2, 0),
    "Darius":        (1, 0, 0, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 2, 2, 2, 0, 0),
    "Diana":         (2, 0, 0, 3, 2, 2, 1, 3, 0, 2, 1, 3, 1, 3, 2, 2, 1, 0),
    "Dr. Mundo":     (0, 0, 1, 1, 3, 2, 0, 1, 0, 2, 3, 1, 3, 1, 2, 0, 0, 0),
    "Draven":        (0, 0, 1, 0, 0, 3, 1, 1, 2, 2, 0, 0, 0, 3, 1, 0, 1, 0),
    "Ekko":          (1, 1, 1, 3, 2, 1, 1, 3, 0, 1, 1, 3, 1, 2, 2, 2, 2, 0),
    "Elise":         (2, 0, 1, 2, 0, 1, 1, 2, 0, 2, 1, 3, 1, 2, 1, 3, 1, 0),
    "Evelynn":       (0, 0, 0, 3, 2, 1, 1, 2, 0, 1, 0, 3, 1, 3, 1, 3, 1, 0),
    "Ezreal":        (0, 0, 3, 0, 2, 2, 0, 2, 2, 1, 0, 1, 0, 2, 1, 0, 3, 0),
    "Fiddlesticks":  (3, 1, 1, 1, 2, 1, 2, 1, 2, 1, 0, 3, 1, 3, 1, 2, 0, 0),
    "Fiora":         (0, 1, 0, 1, 3, 2, 1, 2, 0, 3, 1, 0, 2, 3, 3, 0, 1, 0),
    "Fizz":          (1, 0, 0, 3, 2, 1, 2, 3, 0, 1, 0, 3, 0, 3, 2, 2, 1, 0),
    "Galio":         (2, 2, 1, 1, 2, 2, 3, 1, 0, 1, 3, 3, 1, 1, 1, 2, 0, 0),
    "Gangplank":     (0, 1, 2, 0, 3, 2, 0, 1, 0, 1, 1, 1, 2, 3, 3, 0, 1, 0),
    "Garen":         (0, 0, 0, 1, 2, 2, 1, 1, 0, 2, 2, 0, 3, 2, 2, 0, 0, 0),
    "Gnar":          (2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 0, 2, 2, 1, 2, 0),
    "Gragas":        (2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 2, 3, 2, 2, 2, 2, 2, 0),
    "Graves":        (0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 1, 0, 0, 2, 2, 0, 1, 0),
    "Gwen":          (0, 1, 0, 2, 3, 1, 0, 2, 0, 2, 1, 3, 2, 3, 3, 0, 1, 0),
    "Hecarim":       (2, 1, 0, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0),
    "Heimerdinger":  (0, 1, 3, 0, 1, 3, 1, 0, 2, 1, 0, 3, 0, 2, 2, 0, 1, 0),
    "Hwei":          (0, 1, 3, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 3, 1, 1, 1, 0),
    "Illaoi":        (1, 0, 0, 0, 2, 3, 1, 0, 0, 2, 2, 0, 2, 2, 3, 1, 0, 0),
    "Irelia":        (1, 0, 0, 2, 2, 2, 1, 3, 0, 2, 1, 1, 1, 2, 3, 1, 2, 0),
    "Ivern":         (0, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 1, 1, 2),
    "Janna":         (0, 3, 1, 0, 2, 1, 2, 1, 2, 1, 0, 2, 0, 0, 0, 0, 2, 3),
    "Jarvan IV":     (3, 1, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 1, 0),
    "Jax":           (1, 1, 0, 1, 3, 2, 1, 2, 0, 3, 2, 2, 0, 3, 3, 0, 1, 0),
    "Jayce":         (1, 1, 3, 1, 1, 3, 1, 1, 2, 1, 1, 1, 0, 2, 2, 1, 2, 0),
    "Jhin":          (0, 1, 2, 0, 2, 2, 1, 0, 3, 1, 0, 0, 0, 2, 1, 1, 2, 0),
    "Jinx":          (0, 1, 2, 0, 3, 2, 1, 1, 3, 2, 0, 0, 0, 3, 1, 0, 3, 0),
    "Kai'Sa":        (0, 0, 1, 2, 3, 1, 0, 1, 2, 2, 0, 2, 0, 3, 1, 1, 2, 0),
    "Kalista":       (1, 2, 1, 0, 1, 3, 0, 2, 2, 2, 0, 0, 0, 2, 1, 1, 3, 0),
    "Karma":         (0, 1, 2, 0, 1, 3, 1, 1, 2, 1, 0, 3, 0, 1, 1, 0, 1, 2),
    "Karthus":       (0, 0, 2, 0, 3, 2, 0, 0, 2, 2, 0, 3, 0, 3, 1, 0, 0, 0),
    "Kassadin":      (0, 0, 0, 2, 3, 0, 0, 3, 0, 1, 1, 3, 0, 3, 2, 1, 2, 0),
    "Katarina":      (0, 0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 3, 0, 3, 2, 1, 1, 0),
    "Kayle":         (0, 2, 1, 0, 3, 1, 0, 1, 2, 2, 0, 2, 1, 3, 3, 0, 3, 1),
    "Kayn":          (1, 0, 0, 3, 2, 2, 1, 3, 0, 2, 1, 1, 1, 2, 2, 3, 1, 0),
    "Kennen":        (3, 1, 1, 2, 2, 2, 3, 1, 2, 1, 1, 3, 0, 2, 1, 1, 1, 0),
    "Kha'Zix":       (0, 0, 1, 3, 1, 1, 1, 3, 0, 2, 1, 0, 1, 3, 1, 3, 1, 0),
    "Kindred":       (0, 2, 1, 1, 3, 2, 0, 2, 2, 3, 0, 1, 1, 2, 1, 0, 2, 1),
    "Kled":          (2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 3, 1, 1, 0),
    "Kog'Maw":       (0, 0, 3, 0, 3, 1, 0, 0, 3, 3, 0, 2, 0, 3, 1, 0, 1, 0),
    "K'Sante":       (2, 2, 1, 1, 2, 2, 2, 2, 0, 1, 3, 1, 1, 1, 1, 2, 1, 0),
    "LeBlanc":       (1, 0, 2, 3, 1, 2, 1, 3, 2, 1, 0, 3, 0, 3, 1, 3, 2, 0),
    "Lee Sin":       (2, 2, 0, 2, 0, 2, 2, 3, 0, 2, 1, 1, 2, 2, 2, 2, 2, 1),
    "Leona":         (3, 3, 0, 1, 1, 1, 3, 1, 0, 1, 3, 3, 0, 0, 0, 2, 0, 0),
    "Lillia":        (1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 3, 1, 2, 1, 1, 3, 0),
    "Lissandra":     (2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0),
    "Locke":         (0, 0, 1, 3, 2, 2, 0, 3, 0, 1, 1, 3, 1, 2, 3, 3, 2, 0),
    "Lucian":        (0, 0, 2, 1, 0, 3, 0, 2, 2, 1, 0, 0, 0, 2, 1, 0, 3, 0),
    "Lulu":          (0, 3, 1, 0, 2, 2, 1, 1, 2, 1, 0, 3, 0, 0, 0, 0, 1, 3),
    "Lux":           (1, 1, 3, 0, 1, 2, 1, 0, 3, 1, 0, 3, 0, 3, 1, 2, 1, 1),
    "Malphite":      (3, 1, 1, 1, 2, 1, 2, 0, 0, 1, 3, 2, 1, 1, 1, 1, 0, 0),
    "Malzahar":      (1, 1, 1, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 2, 1, 2, 0, 0),
    "Maokai":        (3, 2, 1, 1, 2, 2, 3, 0, 0, 1, 3, 3, 2, 1, 1, 2, 0, 0),
    "Master Yi":     (0, 0, 0, 2, 3, 2, 0, 2, 0, 3, 0, 1, 2, 3, 3, 0, 1, 0),
    "Mel":           (0, 2, 3, 0, 3, 2, 1, 1, 2, 1, 1, 3, 0, 3, 1, 0, 2, 0),
    "Milio":         (0, 3, 1, 0, 2, 2, 0, 1, 2, 1, 0, 3, 0, 0, 0, 0, 1, 3),
    "Miss Fortune":  (0, 0, 2, 0, 2, 3, 1, 0, 2, 1, 0, 1, 0, 3, 1, 0, 1, 0),
    "Wukong":        (3, 1, 0, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 1, 1, 0),
    "Mordekaiser":   (2, 1, 0, 1, 2, 2, 1, 0, 0, 1, 3, 3, 2, 1, 2, 2, 0, 0),
    "Morgana":       (1, 2, 2, 0, 2, 2, 2, 0, 2, 1, 0, 3, 1, 1, 1, 2, 1, 1),
    "Naafiri":       (1, 0, 0, 3, 1, 2, 1, 2, 0, 1, 1, 1, 1, 3, 2, 3, 1, 0),
    "Nami":          (1, 2, 2, 0, 2, 2, 1, 1, 2, 1, 0, 3, 0, 1, 0, 1, 1, 3),
    "Nasus":         (0, 1, 0, 1, 3, 1, 1, 0, 0, 2, 2, 2, 2, 2, 3, 0, 0, 0),
    "Nautilus":      (3, 2, 0, 2, 2, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 2, 0, 0),
    "Neeko":         (2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0),
    "Nidalee":       (0, 1, 3, 1, 1, 2, 0, 2, 2, 2, 0, 3, 2, 2, 1, 2, 2, 1),
    "Nilah":         (1, 1, 0, 2, 3, 1, 1, 2, 0, 2, 0, 1, 1, 3, 2, 0, 2, 0),
    "Nocturne":      (2, 1, 0, 2, 1, 2, 2, 2, 0, 2, 1, 1, 0, 2, 2, 2, 1, 0),
    "Nunu & Willump": (3, 2, 0, 1, 2, 2, 2, 2, 0, 2, 3, 3, 2, 1, 0, 1, 1, 0),
    "Olaf":          (0, 0, 0, 1, 1, 3, 0, 1, 0, 2, 2, 0, 2, 2, 3, 0, 1, 0),
    "Orianna":       (1, 1, 2, 0, 2, 3, 2, 0, 2, 1, 0, 3, 1, 2, 1, 1, 1, 2),
    "Ornn":          (3, 2, 0, 1, 2, 2, 3, 0, 0, 2, 3, 2, 1, 1, 2, 1, 0, 0),
    "Pantheon":      (1, 1, 2, 1, 0, 3, 1, 1, 0, 1, 1, 0, 0, 2, 2, 2, 1, 0),
    "Poppy":         (2, 3, 0, 1, 2, 2, 3, 1, 0, 1, 3, 1, 0, 1, 1, 1, 1, 0),
    "Pyke":          (2, 1, 0, 2, 0, 2, 2, 3, 0, 1, 0, 0, 2, 2, 1, 3, 2, 0),
    "Qiyana":        (1, 0, 0, 3, 1, 2, 1, 3, 0, 1, 0, 1, 0, 3, 1, 3, 2, 0),
    "Quinn":         (0, 1, 2, 1, 1, 2, 1, 3, 2, 1, 0, 0, 0, 2, 2, 1, 2, 0),
    "Rakan":         (3, 3, 0, 2, 2, 1, 3, 3, 0, 1, 1, 3, 1, 0, 0, 1, 2, 2),
    "Rammus":        (2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 3, 2, 0, 1, 1, 1, 1, 0),
    "Rek'Sai":       (2, 0, 0, 1, 0, 2, 1, 3, 0, 1, 2, 1, 1, 1, 1, 1, 1, 0),
    "Rell":          (3, 2, 0, 1, 2, 1, 3, 2, 0, 1, 3, 3, 1, 0, 0, 2, 0, 1),
    "Renata Glasc":  (1, 2, 1, 0, 2, 2, 2, 0, 2, 1, 0, 3, 0, 0, 0, 1, 0, 2),
    "Renekton":      (2, 1, 0, 1, 0, 3, 2, 2, 0, 1, 2, 0, 2, 2, 2, 1, 1, 0),
    "Rengar":        (1, 0, 0, 3, 1, 1, 1, 3, 0, 2, 0, 1, 2, 3, 2, 3, 1, 0),
    "Riven":         (2, 0, 0, 2, 1, 2, 1, 3, 0, 2, 1, 0, 1, 2, 2, 1, 1, 0),
    "Rumble":        (2, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 3, 1, 2, 1, 0, 1, 0),
    "Ryze":          (0, 1, 1, 0, 3, 2, 1, 1, 2, 2, 0, 3, 0, 2, 2, 1, 1, 0),
    "Samira":        (0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 0, 0, 2, 3, 1, 0, 2, 0),
    "Sejuani":       (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 3, 0, 1, 1, 2, 1, 0),
    "Senna":         (0, 1, 3, 0, 3, 1, 1, 1, 3, 1, 0, 2, 1, 2, 1, 1, 1, 2),
    "Seraphine":     (1, 2, 3, 0, 3, 3, 1, 0, 3, 1, 0, 3, 0, 1, 0, 1, 1, 3),
    "Sett":          (2, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 2, 2, 2, 2, 1, 0),
    "Shaco":         (0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 0, 1, 0, 2, 1, 2, 1, 0),
    "Shen":          (2, 3, 0, 1, 2, 2, 2, 1, 0, 1, 3, 2, 1, 1, 2, 1, 0, 2),
    "Shyvana":       (1, 0, 0, 1, 2, 2, 0, 2, 0, 3, 2, 2, 1, 2, 2, 0, 1, 0),
    "Singed":        (1, 1, 0, 0, 2, 2, 1, 2, 0, 1, 2, 3, 1, 1, 3, 0, 3, 0),
    "Sion":          (2, 1, 0, 1, 3, 3, 2, 1, 0, 2, 3, 2, 2, 1, 3, 0, 0, 0),
    "Sivir":         (0, 1, 1, 0, 2, 3, 0, 1, 2, 2, 0, 0, 0, 2, 1, 0, 2, 0),
    "Skarner":       (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 2, 1, 1, 1, 2, 0, 0),
    "Smolder":       (0, 0, 2, 0, 3, 1, 0, 1, 2, 1, 0, 1, 0, 3, 1, 0, 1, 0),
    "Sona":          (0, 2, 2, 0, 3, 1, 1, 0, 2, 1, 0, 3, 0, 0, 0, 0, 1, 3),
    "Soraka":        (0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 0, 3, 0, 0, 0, 0, 1, 3),
    "Swain":         (1, 1, 2, 0, 2, 2, 1, 0, 2, 1, 2, 3, 2, 2, 1, 1, 0, 0),
    "Sylas":         (1, 0, 0, 3, 2, 2, 1, 3, 0, 1, 2, 3, 2, 3, 2, 1, 2, 0),
    "Syndra":        (1, 1, 2, 1, 3, 3, 1, 0, 2, 1, 0, 3, 0, 3, 1, 2, 1, 0),
    "Tahm Kench":    (1, 2, 0, 0, 2, 2, 2, 1, 0, 1, 3, 2, 2, 1, 2, 1, 0, 0),
    "Taliyah":       (1, 1, 2, 0, 2, 3, 2, 2, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0),
    "Talon":         (0, 0, 1, 3, 1, 2, 0, 3, 0, 1, 0, 0, 0, 3, 2, 3, 1, 0),
    "Taric":         (1, 3, 0, 0, 2, 1, 2, 0, 0, 1, 3, 3, 2, 0, 0, 0, 0, 3),
    "Teemo":         (0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 0, 3, 0, 2, 2, 0, 2, 0),
    "Thresh":        (3, 3, 1, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 0, 0, 3, 1, 1),
    "Tristana":      (0, 1, 1, 2, 2, 2, 1, 2, 3, 2, 0, 1, 0, 3, 2, 0, 2, 0),
    "Trundle":       (1, 1, 0, 1, 2, 2, 1, 0, 0, 3, 3, 1, 2, 1, 3, 0, 1, 0),
    "Tryndamere":    (0, 0, 0, 2, 3, 2, 0, 1, 0, 3, 1, 0, 3, 3, 3, 0, 1, 0),
    "Twisted Fate":  (1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0, 3, 0, 2, 2, 2, 1, 0),
    "Twitch":        (0, 0, 1, 1, 3, 1, 1, 1, 2, 2, 0, 1, 0, 3, 1, 1, 1, 0),
    "Udyr":          (1, 1, 0, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 3, 0, 2, 0),
    "Urgot":         (2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 1, 0, 0),
    "Varus":         (1, 1, 3, 0, 2, 3, 1, 0, 3, 2, 0, 2, 0, 2, 1, 1, 2, 0),
    "Vayne":         (0, 1, 0, 1, 3, 0, 1, 2, 2, 3, 0, 1, 0, 3, 2, 0, 3, 0),
    "Veigar":        (1, 2, 1, 0, 3, 1, 2, 0, 2, 1, 0, 3, 0, 3, 1, 2, 1, 0),
    "Vel'Koz":       (0, 1, 3, 0, 2, 2, 1, 0, 3, 1, 0, 3, 0, 3, 1, 0, 0, 0),
    "Vex":           (1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0),
    "Vi":            (3, 1, 0, 3, 2, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 3, 1, 0),
    "Viego":         (1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 1, 1, 2, 3, 2, 1, 1, 0),
    "Viktor":        (1, 1, 3, 0, 3, 3, 1, 0, 2, 1, 0, 3, 0, 3, 1, 0, 1, 0),
    "Vladimir":      (0, 0, 1, 1, 3, 1, 0, 1, 2, 1, 1, 3, 3, 3, 2, 0, 1, 0),
    "Volibear":      (2, 1, 0, 1, 2, 2, 2, 1, 0, 2, 3, 2, 2, 1, 2, 1, 1, 0),
    "Warwick":       (2, 1, 0, 1, 1, 2, 2, 1, 0, 2, 2, 2, 3, 2, 2, 2, 1, 0),
    "Xayah":         (0, 1, 1, 1, 3, 2, 1, 1, 2, 2, 0, 0, 1, 3, 1, 1, 3, 0),
    "Xerath":        (0, 0, 3, 0, 2, 3, 1, 0, 3, 1, 0, 3, 0, 3, 1, 0, 1, 0),
    "Xin Zhao":      (3, 1, 0, 2, 1, 2, 1, 2, 0, 2, 2, 0, 2, 2, 2, 1, 1, 0),
    "Yasuo":         (2, 1, 0, 2, 3, 2, 1, 3, 0, 2, 1, 0, 1, 3, 2, 1, 2, 0),
    "Yone":          (2, 0, 0, 3, 3, 2, 1, 3, 0, 2, 1, 1, 1, 3, 2, 1, 2, 0),
    "Yorick":        (0, 1, 0, 0, 3, 3, 1, 0, 0, 3, 2, 2, 2, 2, 3, 0, 0, 0),
    "Yunara":        (0, 1, 1, 2, 3, 2, 0, 2, 2, 3, 0, 1, 1, 3, 0, 1, 2, 0),
    "Yuumi":         (0, 2, 1, 0, 3, 1, 1, 1, 2, 1, 0, 3, 0, 0, 0, 0, 1, 3),
    "Zaahen":        (2, 1, 0, 2, 2, 1, 1, 1, 1, 2, 2, 0, 3, 1, 3, 2, 1, 0),
    "Zac":           (3, 2, 0, 1, 2, 1, 2, 3, 0, 1, 3, 2, 3, 1, 0, 1, 1, 0),
    "Zed":           (0, 0, 1, 3, 1, 2, 0, 3, 0, 1, 0, 0, 0, 3, 2, 3, 1, 0),
    "Zeri":          (0, 0, 1, 0, 3, 2, 0, 3, 2, 2, 0, 1, 0, 3, 1, 0, 3, 0),
    "Ziggs":         (0, 1, 3, 0, 2, 3, 1, 1, 3, 2, 0, 3, 0, 3, 2, 0, 1, 0),
    "Zilean":        (0, 3, 1, 0, 2, 2, 1, 1, 2, 1, 0, 3, 0, 1, 0, 0, 1, 2),
    "Zoe":           (1, 1, 3, 1, 2, 2, 2, 1, 3, 1, 0, 3, 0, 3, 1, 3, 1, 0),
    "Zyra":          (1, 2, 3, 0, 2, 2, 2, 0, 3, 1, 0, 3, 0, 2, 1, 1, 1, 0),
}

# Tagged, but from thin knowledge: the numbers are a best reading of the kit,
# not a confident one.
LOW_CONFIDENCE = {
    "Mel": "released after my reliable knowledge cutoff; reviewed by the "
           "project owner, who corrected peel 3 -> 2",
    "Yunara": "released after my reliable knowledge cutoff; vector supplied "
              "and neighbour-checked by the project owner",
    "Locke": "kit given as 'AP mid, melee, mobile assassin'; vector supplied "
             "and neighbour-checked (closest: Akali 2.00)",
    "Zaahen": "kit given as 'AD melee top bruiser'; vector supplied and "
              "neighbour-checked (closest: Sett 2.65)",
}

# No champion is left untagged; the mechanism stays because the next new
# champion will need it.
UNKNOWN: dict[str, str] = {}

NOTES = {
    "Thresh": "peel 3 and engage 3: hook starts fights, lantern+flay ends "
              "them. healing 1 is the lantern shield, not a heal",
    "Kayle": "peel 2 because R can be cast on an ally, kiting 3 because R on "
             "herself plus range is what keeps her alive - the two are "
             "different jobs and used to share a column",
    "Xayah": "kiting 3, peel 1: R is self-only, so she saves herself far "
             "better than she saves anyone else",
    "Yuumi": "mobility 1 - she moves by attaching, not by dashing; "
             "magic_share 3 with damage_output 0 means 'magic type, no volume'",
    "Janna": "magic_share 2 but damage_output 0: the ratio is meaningless "
             "when nothing is dealt, which is why the axes are separate",
    "Soraka": "healing 3, sustain 0 - she heals everyone except herself",
    "Warwick": "sustain 3, healing 0 - the classic case that forced the split",
    "Master Yi": "magic_share 1: Wuju Style is true damage, which counts as "
                 "neither physical nor magic",
    "Kennen": "engage 3 is R-only, but R alone is enough to start a fight",
    "Kassadin": "lane_priority 0 is the point of the champion, not an "
                "oversight",
    "Blitzcrank": "dive 1 but pick 3: the hook catches a straggler, it does "
                  "not get him through a frontline",
    "Malphite": "engage 3 but pick 1: R starts a teamfight, it cannot isolate "
                "one target",
    "Singed": "kiting 3 with mobility 2: he does not dash, he runs, and that "
              "is the whole champion",
}


def build(out_path: Path = OUT_PATH) -> dict:
    names = {int(k): v for k, v in
             json.loads(NAMES_PATH.read_text(encoding="utf-8"))["names"].items()}
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    riot_to_idx = {int(k): int(v) for k, v in mapping["riot_to_idx"].items()}
    patch = mapping.get("patch")

    name_to_riot = {name: riot for riot, name in names.items()}
    missing = sorted(set(name_to_riot) - set(TABLE) - set(UNKNOWN))
    if missing:
        raise SystemExit("champions with no row: %s" % missing)
    extra = sorted(set(TABLE) - set(name_to_riot))
    if extra:
        raise SystemExit("rows for champions not in the mapping: %s" % extra)

    k = len(AXES)
    tank_i, dmg_i = AXES.index(next(a for a in AXES if a[0] == "tankiness")), \
        AXES.index(next(a for a in AXES if a[0] == "damage_output"))
    for name, row in TABLE.items():
        if len(row) != k:
            raise SystemExit("%s has %d values, expected %d" % (name, len(row), k))
        bad = [v for v in row if not isinstance(v, int) or not 0 <= v <= 3]
        if bad:
            raise SystemExit("%s has out-of-range values: %s" % (name, bad))
        # Rule 1 in the AXES comment: a tank is never the team's damage.
        if row[tank_i] == 3 and row[dmg_i] > 1:
            raise SystemExit(
                "%s is tagged as a tank (tankiness 3) with damage_output %d. "
                "A tank caps at 1; if it really is a damage threat, the tag "
                "to change is tankiness (2 = bruiser), not this one."
                % (name, row[dmg_i]))

    champs = {}
    for name, row in TABLE.items():
        riot = name_to_riot[name]
        champs[str(riot)] = {
            "name": name,
            "idx": riot_to_idx[riot],
            "attrs": list(row),
            "confidence": "low" if name in LOW_CONFIDENCE else "high",
        }
    for name, why in UNKNOWN.items():
        riot = name_to_riot[name]
        champs[str(riot)] = {
            "name": name,
            "idx": riot_to_idx[riot],
            "attrs": None,
            "confidence": "unknown",
            "note": why,
        }

    payload = {
        "patch": patch,
        "num_champs": int(mapping["num_champs"]),
        "axes": [a[0] for a in AXES],
        "scale": "ordinal 0..3; divide by 3 before use",
        "anchors": {a[0]: {"0": a[1], "1": a[2], "2": a[3], "3": a[4]}
                    for a in AXES},
        "provenance": "hand-tagged from kit knowledge at patch %s; "
                      "no model queried at build time" % patch,
        "notes": NOTES,
        "unknown": {n: w for n, w in UNKNOWN.items()},
        "low_confidence": {n: w for n, w in LOW_CONFIDENCE.items()},
        "champions": champs,
    }
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    return payload


def matrix(payload: dict, num_champs: int | None = None):
    """-> (names, rows) with rows[i] the attribute list for idx i, or None."""
    n = num_champs or payload["num_champs"]
    names = [None] * n
    rows = [None] * n
    for entry in payload["champions"].values():
        names[entry["idx"]] = entry["name"]
        rows[entry["idx"]] = entry["attrs"]
    return names, rows


if __name__ == "__main__":
    out = build()
    tagged = sum(1 for c in out["champions"].values() if c["attrs"])
    print("wrote %s: %d champions, %d tagged, %d unknown, %d axes"
          % (OUT_PATH.name, len(out["champions"]), tagged,
             len(UNKNOWN), len(AXES)))
