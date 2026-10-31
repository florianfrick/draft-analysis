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
reviewed, diffed and corrected. ``confidence`` records where that knowledge
is thin.

Two revisions after review (2026-10-30):

* ``magic_damage`` became ``magic_share`` and a separate ``damage_output``
  axis was added. The original column was ambiguous and, read as a damage
  level, wrong: Yuumi scored 3 and deals almost nothing. A champion's damage
  *type* and its damage *volume* are independent, and drafting cares about
  both ("we need magic damage" is not satisfied by a healer).
* ``side_laning`` and ``pick_potential`` were added: holding a side lane
  alone, and isolating one target, are two things the original axes could
  not express — the first was missing entirely, and pick potential is
  genuinely different from engage (Blitzcrank dives 1, picks 3; Malphite
  engages 3, picks 1).
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NAMES_PATH = BASE_DIR / "solo queue" / "process matchdata" / "champ_names.json"
MAPPING_PATH = BASE_DIR / "solo queue" / "process matchdata" / "champ_mapping.json"
OUT_PATH = BASE_DIR / "champ_attrs.json"

# (name, 0-anchor, 1, 2, 3-anchor). Higher is always "more of the thing".
AXES = [
    ("engage", "no way to start a fight",
     "situational or single-target initiation",
     "reliable initiation, usually on one target",
     "reliable AoE fight-start (Malphite R, Rakan W/R, Rell R)"),
    ("peel", "nothing to protect a carry with",
     "one small tool (a slow, a short shield)",
     "solid: a displacement, a real shield, a zone",
     "elite disengage that cancels a dive (Janna Q/R, Poppy W)"),
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
    ("sustain", "no healing or shielding for the team",
     "self-sustain only, or a token shield",
     "meaningful healing/shielding for allies",
     "elite enchanter throughput (Soraka, Yuumi, Sona, Lulu)"),
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
]

# name -> (engage, peel, poke, dive, scaling, lane_priority, hard_cc,
#          mobility, range, objective_speed, tankiness, magic_share,
#          sustain, damage_output, side_laning, pick_potential)
TABLE = {
    "Aatrox":        (1, 0, 1, 1, 1, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 0),
    "Ahri":          (1, 0, 2, 2, 2, 2, 1, 3, 2, 1, 0, 3, 0, 2, 1, 3),
    "Akali":         (0, 0, 1, 3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 3, 2, 2),
    "Akshan":        (0, 0, 2, 2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 2, 2, 1),
    "Alistar":       (3, 2, 0, 1, 1, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 2),
    "Ambessa":       (1, 0, 0, 2, 2, 2, 0, 3, 0, 2, 2, 0, 1, 2, 2, 1),
    "Amumu":         (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 0, 2),
    "Anivia":        (0, 2, 2, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 2, 1, 1),
    "Annie":         (1, 1, 1, 1, 2, 2, 2, 0, 2, 1, 0, 3, 0, 2, 1, 2),
    "Aphelios":      (0, 1, 1, 0, 3, 1, 1, 0, 2, 2, 0, 0, 0, 3, 1, 0),
    "Ashe":          (2, 1, 2, 0, 2, 2, 2, 0, 2, 1, 0, 0, 0, 1, 1, 2),
    "Aurelion Sol":  (0, 1, 1, 1, 3, 2, 1, 1, 2, 1, 0, 3, 0, 3, 1, 0),
    "Aurora":        (1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 3, 0, 2, 1, 2),
    "Azir":          (1, 2, 2, 1, 3, 2, 1, 1, 2, 2, 0, 3, 0, 2, 1, 1),
    "Bard":          (2, 2, 1, 0, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2),
    "Bel'Veth":      (1, 0, 0, 2, 2, 2, 1, 2, 0, 3, 1, 1, 1, 3, 2, 1),
    "Blitzcrank":    (3, 1, 0, 1, 1, 1, 3, 1, 0, 1, 2, 2, 0, 0, 0, 3),
    "Brand":         (0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 0, 3, 0, 3, 1, 1),
    "Braum":         (1, 3, 1, 0, 2, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 1),
    "Briar":         (1, 0, 0, 3, 1, 2, 2, 2, 0, 2, 1, 0, 1, 2, 2, 2),
    "Caitlyn":       (0, 1, 3, 0, 1, 3, 1, 1, 3, 2, 0, 0, 0, 2, 1, 1),
    "Camille":       (1, 1, 0, 2, 2, 2, 2, 3, 0, 2, 2, 0, 1, 2, 3, 2),
    "Cassiopeia":    (0, 1, 2, 0, 3, 2, 1, 1, 2, 2, 0, 3, 0, 3, 1, 1),
    "Cho'Gath":      (1, 1, 1, 0, 3, 2, 2, 0, 0, 2, 3, 2, 1, 2, 2, 2),
    "Corki":         (0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 1, 1),
    "Darius":        (1, 0, 0, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 2, 2, 2),
    "Diana":         (2, 0, 0, 3, 2, 2, 1, 3, 0, 2, 1, 3, 1, 3, 2, 2),
    "Dr. Mundo":     (0, 0, 1, 1, 3, 2, 0, 1, 0, 2, 3, 1, 2, 2, 2, 0),
    "Draven":        (0, 0, 1, 0, 0, 3, 1, 1, 2, 2, 0, 0, 0, 3, 1, 0),
    "Ekko":          (1, 1, 1, 3, 2, 1, 1, 3, 0, 1, 1, 3, 1, 2, 2, 2),
    "Elise":         (2, 0, 1, 2, 0, 1, 1, 2, 0, 2, 1, 3, 1, 2, 1, 3),
    "Evelynn":       (0, 0, 0, 3, 2, 1, 1, 2, 0, 1, 0, 3, 1, 3, 1, 3),
    "Ezreal":        (0, 0, 3, 0, 2, 2, 0, 2, 2, 1, 0, 1, 0, 2, 1, 0),
    "Fiddlesticks":  (3, 1, 1, 1, 2, 1, 2, 1, 2, 1, 0, 3, 1, 3, 1, 2),
    "Fiora":         (0, 1, 0, 1, 3, 2, 1, 2, 0, 3, 1, 0, 2, 3, 3, 0),
    "Fizz":          (1, 0, 0, 3, 2, 1, 2, 3, 0, 1, 0, 3, 0, 3, 2, 2),
    "Galio":         (2, 2, 1, 1, 2, 2, 3, 1, 0, 1, 3, 3, 1, 1, 1, 2),
    "Gangplank":     (0, 1, 2, 0, 3, 2, 0, 1, 0, 1, 1, 1, 1, 3, 3, 0),
    "Garen":         (0, 0, 0, 1, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 2, 0),
    "Gnar":          (2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 0, 2, 2, 1),
    "Gragas":        (2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 2, 3, 2, 2, 2, 2),
    "Graves":        (0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 1, 0, 0, 2, 2, 0),
    "Gwen":          (0, 1, 0, 2, 3, 1, 0, 2, 0, 2, 1, 3, 1, 3, 3, 0),
    "Hecarim":       (2, 1, 0, 2, 2, 2, 2, 3, 0, 2, 2, 2, 1, 2, 1, 2),
    "Heimerdinger":  (0, 1, 3, 0, 1, 3, 1, 0, 2, 1, 0, 3, 0, 2, 2, 0),
    "Hwei":          (0, 1, 3, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 3, 1, 1),
    "Illaoi":        (1, 0, 0, 0, 2, 3, 1, 0, 0, 2, 2, 0, 1, 2, 3, 1),
    "Irelia":        (1, 0, 0, 2, 2, 2, 1, 3, 0, 2, 1, 1, 1, 2, 3, 1),
    "Ivern":         (0, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1),
    "Janna":         (0, 3, 1, 0, 2, 1, 2, 1, 2, 1, 0, 2, 3, 0, 0, 0),
    "Jarvan IV":     (3, 1, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2),
    "Jax":           (1, 1, 0, 1, 3, 2, 1, 2, 0, 3, 2, 2, 0, 3, 3, 0),
    "Jayce":         (1, 1, 3, 1, 1, 3, 1, 1, 2, 1, 1, 1, 0, 2, 2, 1),
    "Jhin":          (0, 1, 2, 0, 2, 2, 1, 0, 3, 1, 0, 0, 0, 2, 1, 1),
    "Jinx":          (0, 1, 2, 0, 3, 2, 1, 1, 3, 2, 0, 0, 0, 3, 1, 0),
    "Kai'Sa":        (0, 0, 1, 2, 3, 1, 0, 1, 2, 2, 0, 2, 0, 3, 1, 1),
    "Kalista":       (1, 2, 1, 0, 1, 3, 0, 2, 2, 2, 0, 0, 0, 2, 1, 1),
    "Karma":         (0, 1, 2, 0, 1, 3, 1, 1, 2, 1, 0, 3, 2, 1, 1, 0),
    "Karthus":       (0, 0, 2, 0, 3, 2, 0, 0, 2, 2, 0, 3, 0, 3, 1, 0),
    "Kassadin":      (0, 0, 0, 2, 3, 0, 0, 3, 0, 1, 1, 3, 0, 3, 2, 1),
    "Katarina":      (0, 0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 3, 0, 3, 2, 1),
    "Kayle":         (0, 2, 1, 0, 3, 1, 0, 1, 2, 2, 0, 2, 1, 3, 3, 0),
    "Kayn":          (1, 0, 0, 3, 2, 2, 1, 3, 0, 2, 1, 1, 1, 2, 2, 3),
    "Kennen":        (3, 1, 1, 2, 2, 2, 3, 1, 2, 1, 1, 3, 0, 2, 1, 1),
    "Kha'Zix":       (0, 0, 1, 3, 1, 1, 1, 3, 0, 2, 1, 0, 1, 3, 1, 3),
    "Kindred":       (0, 2, 1, 1, 3, 2, 0, 2, 2, 3, 0, 1, 1, 2, 1, 0),
    "Kled":          (2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 3, 1),
    "Kog'Maw":       (0, 0, 3, 0, 3, 1, 0, 0, 3, 3, 0, 2, 0, 3, 1, 0),
    "K'Sante":       (2, 2, 1, 1, 2, 2, 2, 2, 0, 1, 3, 1, 1, 1, 1, 2),
    "LeBlanc":       (1, 0, 2, 3, 1, 2, 1, 3, 2, 1, 0, 3, 0, 3, 1, 3),
    "Lee Sin":       (2, 2, 0, 2, 0, 2, 2, 3, 0, 2, 1, 1, 1, 2, 2, 2),
    "Leona":         (3, 2, 0, 1, 1, 1, 3, 1, 0, 1, 3, 3, 0, 0, 0, 2),
    "Lillia":        (1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 3, 1, 2, 1, 1),
    "Lissandra":     (2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 0, 3, 0, 2, 1, 2),
    "Locke":         (1, 0, 1, 3, 1, 2, 1, 3, 0, 1, 0, 3, 0, 3, 2, 3),
    "Lucian":        (0, 0, 2, 1, 0, 3, 0, 2, 2, 1, 0, 0, 0, 2, 1, 0),
    "Lulu":          (0, 3, 1, 0, 2, 2, 1, 1, 2, 1, 0, 3, 3, 0, 0, 0),
    "Lux":           (1, 1, 3, 0, 1, 2, 1, 0, 3, 1, 0, 3, 1, 3, 1, 2),
    "Malphite":      (3, 1, 1, 1, 2, 1, 2, 0, 0, 1, 3, 2, 0, 1, 1, 1),
    "Malzahar":      (1, 1, 1, 0, 2, 3, 2, 0, 2, 1, 0, 3, 0, 2, 1, 2),
    "Maokai":        (3, 2, 1, 1, 2, 2, 3, 0, 0, 1, 3, 3, 1, 1, 1, 2),
    "Master Yi":     (0, 0, 0, 2, 3, 2, 0, 2, 0, 3, 0, 1, 1, 3, 3, 0),
    "Mel":           (0, 1, 2, 0, 2, 2, 1, 1, 2, 1, 0, 3, 1, 2, 1, 1),
    "Milio":         (0, 3, 1, 0, 2, 2, 0, 1, 2, 1, 0, 3, 3, 0, 0, 0),
    "Miss Fortune":  (0, 0, 2, 0, 2, 3, 1, 0, 2, 1, 0, 1, 0, 3, 1, 0),
    "Wukong":        (3, 1, 0, 2, 2, 2, 2, 2, 0, 1, 2, 0, 0, 2, 2, 1),
    "Mordekaiser":   (2, 1, 0, 1, 2, 2, 1, 0, 0, 1, 3, 3, 1, 2, 2, 2),
    "Morgana":       (1, 2, 2, 0, 2, 2, 2, 0, 2, 1, 0, 3, 1, 1, 1, 2),
    "Naafiri":       (1, 0, 0, 3, 1, 2, 1, 2, 0, 1, 1, 1, 0, 3, 2, 3),
    "Nami":          (1, 2, 2, 0, 2, 2, 1, 1, 2, 1, 0, 3, 3, 1, 0, 1),
    "Nasus":         (0, 1, 0, 1, 3, 1, 1, 0, 0, 2, 2, 2, 1, 2, 3, 0),
    "Nautilus":      (3, 2, 0, 2, 2, 1, 3, 1, 0, 1, 3, 2, 0, 0, 0, 2),
    "Neeko":         (2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2),
    "Nidalee":       (0, 1, 3, 1, 1, 2, 0, 2, 2, 2, 0, 3, 1, 2, 1, 2),
    "Nilah":         (1, 1, 0, 2, 3, 1, 1, 2, 0, 2, 0, 1, 1, 3, 2, 0),
    "Nocturne":      (2, 1, 0, 2, 1, 2, 2, 2, 0, 2, 1, 1, 0, 2, 2, 2),
    "Nunu & Willump": (3, 2, 0, 1, 2, 2, 2, 2, 0, 2, 3, 3, 2, 1, 0, 1),
    "Olaf":          (0, 0, 0, 1, 1, 3, 0, 1, 0, 2, 2, 0, 2, 2, 3, 0),
    "Orianna":       (1, 1, 2, 0, 2, 3, 2, 0, 2, 1, 0, 3, 1, 2, 1, 1),
    "Ornn":          (3, 2, 0, 1, 2, 2, 3, 0, 0, 2, 3, 2, 0, 1, 2, 1),
    "Pantheon":      (1, 1, 2, 1, 0, 3, 1, 1, 0, 1, 1, 0, 0, 2, 2, 2),
    "Poppy":         (2, 3, 0, 1, 2, 2, 3, 1, 0, 1, 3, 1, 0, 1, 1, 1),
    "Pyke":          (2, 1, 0, 2, 0, 2, 2, 3, 0, 1, 0, 0, 1, 2, 1, 3),
    "Qiyana":        (1, 0, 0, 3, 1, 2, 1, 3, 0, 1, 0, 1, 0, 3, 1, 3),
    "Quinn":         (0, 1, 2, 1, 1, 2, 1, 3, 2, 1, 0, 0, 0, 2, 2, 1),
    "Rakan":         (3, 2, 0, 2, 2, 1, 3, 3, 0, 1, 1, 3, 2, 0, 0, 1),
    "Rammus":        (2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 3, 2, 0, 1, 1, 1),
    "Rek'Sai":       (2, 0, 0, 1, 0, 2, 1, 3, 0, 1, 2, 1, 1, 1, 1, 1),
    "Rell":          (3, 2, 0, 1, 2, 1, 3, 2, 0, 1, 3, 3, 1, 0, 0, 2),
    "Renata Glasc":  (1, 2, 1, 0, 2, 2, 2, 0, 2, 1, 0, 3, 2, 0, 0, 1),
    "Renekton":      (2, 1, 0, 1, 0, 3, 2, 2, 0, 1, 2, 0, 1, 2, 2, 1),
    "Rengar":        (1, 0, 0, 3, 1, 1, 1, 3, 0, 2, 0, 1, 1, 3, 2, 3),
    "Riven":         (2, 0, 0, 2, 1, 2, 1, 3, 0, 2, 1, 0, 1, 2, 2, 1),
    "Rumble":        (2, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 3, 0, 2, 1, 0),
    "Ryze":          (0, 1, 1, 0, 3, 2, 1, 1, 2, 2, 0, 3, 0, 2, 2, 1),
    "Samira":        (0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 0, 0, 1, 3, 1, 0),
    "Sejuani":       (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 3, 0, 1, 1, 2),
    "Senna":         (0, 1, 3, 0, 3, 1, 1, 1, 3, 1, 0, 2, 2, 2, 1, 1),
    "Seraphine":     (1, 2, 3, 0, 3, 3, 1, 0, 3, 1, 0, 3, 3, 1, 0, 1),
    "Sett":          (2, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 2, 2, 2, 2),
    "Shaco":         (0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 0, 1, 0, 2, 1, 2),
    "Shen":          (2, 3, 0, 1, 2, 2, 2, 1, 0, 1, 3, 2, 2, 1, 2, 1),
    "Shyvana":       (1, 0, 0, 1, 2, 2, 0, 2, 0, 3, 2, 2, 0, 2, 2, 0),
    "Singed":        (1, 1, 0, 0, 2, 2, 1, 2, 0, 1, 2, 3, 1, 1, 3, 0),
    "Sion":          (2, 1, 0, 1, 3, 3, 2, 1, 0, 2, 3, 2, 1, 2, 3, 0),
    "Sivir":         (0, 1, 1, 0, 2, 3, 0, 1, 2, 2, 0, 0, 0, 2, 1, 0),
    "Skarner":       (3, 2, 0, 1, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 1, 2),
    "Smolder":       (0, 0, 2, 0, 3, 1, 0, 1, 2, 1, 0, 1, 0, 3, 1, 0),
    "Sona":          (0, 2, 2, 0, 3, 1, 1, 0, 2, 1, 0, 3, 3, 0, 0, 0),
    "Soraka":        (0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 0, 3, 3, 0, 0, 0),
    "Swain":         (1, 1, 2, 0, 2, 2, 1, 0, 2, 1, 2, 3, 2, 2, 1, 1),
    "Sylas":         (1, 0, 0, 3, 2, 2, 1, 3, 0, 1, 2, 3, 2, 3, 2, 1),
    "Syndra":        (1, 1, 2, 1, 3, 3, 1, 0, 2, 1, 0, 3, 0, 3, 1, 2),
    "Tahm Kench":    (1, 2, 0, 0, 2, 2, 2, 1, 0, 1, 3, 2, 2, 1, 2, 1),
    "Taliyah":       (1, 1, 2, 0, 2, 3, 2, 2, 2, 1, 0, 3, 0, 2, 1, 2),
    "Talon":         (0, 0, 1, 3, 1, 2, 0, 3, 0, 1, 0, 0, 0, 3, 2, 3),
    "Taric":         (1, 3, 0, 0, 2, 1, 2, 0, 0, 1, 3, 3, 2, 0, 0, 0),
    "Teemo":         (0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 0, 3, 0, 2, 2, 0),
    "Thresh":        (3, 3, 1, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 0, 0, 3),
    "Tristana":      (0, 1, 1, 2, 2, 2, 1, 2, 3, 2, 0, 1, 0, 3, 2, 0),
    "Trundle":       (1, 1, 0, 1, 2, 2, 1, 0, 0, 3, 3, 1, 2, 2, 3, 0),
    "Tryndamere":    (0, 0, 0, 2, 3, 2, 0, 1, 0, 3, 1, 0, 2, 3, 3, 0),
    "Twisted Fate":  (1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0, 3, 0, 2, 2, 2),
    "Twitch":        (0, 0, 1, 1, 3, 1, 1, 1, 2, 2, 0, 1, 0, 3, 1, 1),
    "Udyr":          (1, 1, 0, 1, 2, 2, 1, 2, 0, 2, 2, 2, 1, 2, 3, 0),
    "Urgot":         (2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 0, 0, 2, 2, 1),
    "Varus":         (1, 1, 3, 0, 2, 3, 1, 0, 3, 2, 0, 2, 0, 2, 1, 1),
    "Vayne":         (0, 1, 0, 1, 3, 0, 1, 2, 2, 3, 0, 1, 0, 3, 2, 0),
    "Veigar":        (1, 2, 1, 0, 3, 1, 2, 0, 2, 1, 0, 3, 0, 3, 1, 2),
    "Vel'Koz":       (0, 1, 3, 0, 2, 2, 1, 0, 3, 1, 0, 3, 0, 3, 1, 0),
    "Vex":           (1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2),
    "Vi":            (3, 1, 0, 3, 2, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 3),
    "Viego":         (1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 1, 1, 1, 3, 2, 1),
    "Viktor":        (1, 1, 3, 0, 3, 3, 1, 0, 2, 1, 0, 3, 0, 3, 1, 0),
    "Vladimir":      (0, 0, 1, 1, 3, 1, 0, 1, 2, 1, 1, 3, 2, 3, 2, 0),
    "Volibear":      (2, 1, 0, 1, 2, 2, 2, 1, 0, 2, 3, 2, 2, 2, 2, 1),
    "Warwick":       (2, 1, 0, 1, 1, 2, 2, 1, 0, 2, 2, 2, 3, 2, 2, 2),
    "Xayah":         (0, 2, 1, 1, 3, 2, 1, 1, 2, 2, 0, 0, 0, 3, 1, 1),
    "Xerath":        (0, 0, 3, 0, 2, 3, 1, 0, 3, 1, 0, 3, 0, 3, 1, 0),
    "Xin Zhao":      (3, 1, 0, 2, 1, 2, 1, 2, 0, 2, 2, 0, 1, 2, 2, 1),
    "Yasuo":         (2, 1, 0, 2, 3, 2, 1, 3, 0, 2, 1, 0, 0, 3, 2, 1),
    "Yone":          (2, 0, 0, 3, 3, 2, 1, 3, 0, 2, 1, 1, 0, 3, 2, 1),
    "Yorick":        (0, 1, 0, 0, 3, 3, 1, 0, 0, 3, 2, 2, 1, 2, 3, 0),
    "Yunara":        (0, 1, 2, 0, 2, 2, 1, 1, 2, 2, 0, 2, 0, 3, 1, 1),
    "Yuumi":         (0, 2, 1, 0, 3, 1, 1, 1, 2, 1, 0, 3, 3, 0, 0, 0),
    "Zaahen":        (1, 0, 0, 1, 2, 2, 1, 1, 0, 2, 2, 0, 1, 2, 2, 1),
    "Zac":           (3, 2, 0, 1, 2, 1, 2, 3, 0, 1, 3, 2, 2, 1, 0, 1),
    "Zed":           (0, 0, 1, 3, 1, 2, 0, 3, 0, 1, 0, 0, 0, 3, 2, 3),
    "Zeri":          (0, 0, 1, 0, 3, 2, 0, 3, 2, 2, 0, 1, 0, 3, 1, 0),
    "Ziggs":         (0, 1, 3, 0, 2, 3, 1, 1, 3, 2, 0, 3, 0, 3, 2, 0),
    "Zilean":        (0, 3, 1, 0, 2, 2, 1, 1, 2, 1, 0, 3, 2, 1, 0, 0),
    "Zoe":           (1, 1, 3, 1, 2, 2, 2, 1, 3, 1, 0, 3, 0, 3, 1, 3),
    "Zyra":          (1, 2, 3, 0, 2, 2, 2, 0, 3, 1, 0, 3, 0, 2, 1, 1),
}

# Tagged, but from thin knowledge: the numbers are a best reading of the kit,
# not a confident one. Flagged so they can be corrected first.
LOW_CONFIDENCE = {
    "Mel": "released after my reliable knowledge cutoff; read as a ranged mage",
    "Yunara": "released after my reliable knowledge cutoff; read as a marksman",
    "Locke": "kit given as 'AP mid, melee, mobile assassin' - tagged from that "
             "one line, everything except mobility/dive/pick is a guess",
    "Zaahen": "kit given as 'AD melee top bruiser' - tagged from that one "
              "line, everything except damage type is a guess",
}

# No champion is left untagged any more; the mechanism stays because the next
# new champion will need it.
UNKNOWN: dict[str, str] = {}

NOTES = {
    "Thresh": "engage 3 and peel 3: hook starts fights, lantern+flay ends them; "
              "damage_output 0 is the point of the champion",
    "Yuumi": "mobility 1 - she moves by attaching, not by dashing; "
             "magic_share 3 with damage_output 0 means 'magic type, no volume'",
    "Janna": "magic_share 2 but damage_output 0: the ratio is meaningless when "
             "nothing is dealt, which is exactly why the two axes are separate",
    "Master Yi": "magic_share 1: Wuju Style is true damage, which counts as "
                 "neither physical nor magic",
    "Kennen": "engage 3 is R-only, but R alone is enough to start a fight",
    "Kassadin": "lane_priority 0 is the point of the champion, not an oversight",
    "Blitzcrank": "dive 1 but pick 3: the hook catches a straggler, it does not "
                  "get him through a frontline",
    "Malphite": "engage 3 but pick 1: R starts a teamfight, it cannot isolate "
                "one target",
    "Zoe": "pick 3 via the bubble at max range; dive 1 because she has no way "
           "to follow it up herself",
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
    for name, row in TABLE.items():
        if len(row) != k:
            raise SystemExit("%s has %d values, expected %d" % (name, len(row), k))
        bad = [v for v in row if not isinstance(v, int) or not 0 <= v <= 3]
        if bad:
            raise SystemExit("%s has out-of-range values: %s" % (name, bad))

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
