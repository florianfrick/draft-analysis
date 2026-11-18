#!/usr/bin/env python3
"""Render the BLG/Gen.G actual-pick rank animation.

Reads `android/video/blg_geng_rank_data.json` (produced by
`android/tools/rank_walkthrough.py`) and writes `blg_geng_rank_bars.mp4`.
Every pick advances one cumulative state; bar heights are eased with a cubic
smoothstep so transitions glide instead of jumping.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "android/video/blg_geng_rank_data.json"
OUT = ROOT / "android/video/blg_geng_rank_bars.mp4"
FRAMES = ROOT / "android/video/.rank_frames"

W, H = 1280, 720
FPS = 30
STEP_FRAMES = 12
INTRO_FRAMES = 45
OUTRO_FRAMES = 75
GAME_GAP_FRAMES = 8

BG = "#0D1420"
PANEL = "#162235"
GRID = "#27374D"
TEXT = "#F4F7FB"
MUTED = "#94A3B8"
SOFT = "#CBD5E1"

BUCKETS = [
    ("1", "#1"),
    ("2", "#2"),
    ("3", "#3"),
    ("4", "#4"),
    ("5", "#5"),
    ("6-10", "6–10"),
    ("11-20", "11–20"),
    ("21+", "21+"),
]
COLORS = {
    "1": "#FF5D73",
    "2": "#FF9A56",
    "3": "#F6CB5B",
    "4": "#68D6A5",
    "5": "#55A8FF",
    "6-10": "#8B8BFF",
    "11-20": "#B18CFA",
    "21+": "#6B7C93",
}
Y_MAX = 25
PLOT_TOP = 168
PLOT_BOTTOM = 545
BAR_W = 96
BAR_GAP = 27
BAR_X0 = 86


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = ("C:/Windows/Fonts/seguisb.ttf" if bold
            else "C:/Windows/Fonts/segoeui.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


F14 = font(14)
F16 = font(16)
F18 = font(18)
F20 = font(20, bold=True)
F22 = font(22, bold=True)
F26 = font(26, bold=True)
F34 = font(34, bold=True)


def bucket_of(rank: int | None) -> str:
    if rank is None:
        return "21+"
    if rank <= 5:
        return str(rank)
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return "21+"


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_states(a: dict[str, float], b: dict[str, float], t: float):
    return {key: a[key] + (b[key] - a[key]) * t for key in b}


def counts_after(records, n: int) -> dict[str, float]:
    state = {key: 0.0 for key, _ in BUCKETS}
    for rec in records[:n]:
        state[bucket_of(rec["rank"])] += 1.0
    return state


def draw_frame(state: dict[str, float], n: int, records, active_idx: int,
               active_t: float, active_key: str | None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Header -------------------------------------------------------------
    d.text((64, 34), "How often was the real pick the model's suggestion?",
           font=F34, fill=TEXT)
    d.text((66, 82),
           "BLG vs Gen.G · Worlds 2023 · rank before each pick · 50 picks total",
           font=F16, fill=MUTED)

    top5 = sum(state[str(i)] for i in range(1, 6))
    pct = (top5 / n * 100) if n else 0.0
    kx, ky, kw, kh = 986, 34, 230, 62
    d.rounded_rectangle((kx, ky, kx + kw, ky + kh), radius=14, fill=PANEL)
    d.text((kx + 16, ky + 9), "TOP-5 HITS", font=F14, fill=MUTED)
    d.text((kx + 16, ky + 31), f"{int(round(top5))} / {n}   {pct:4.0f}%",
           font=F22, fill=TEXT)

    # Grid ----------------------------------------------------------------
    plot_h = PLOT_BOTTOM - PLOT_TOP
    for tick in range(0, Y_MAX + 1, 5):
        y = PLOT_BOTTOM - (tick / Y_MAX) * plot_h
        d.line((74, y, 1228, y), fill=GRID, width=1)
        d.text((64, y - 9), str(tick), font=F14, fill=MUTED, anchor="ra")

    # Bars ----------------------------------------------------------------
    for i, (key, label) in enumerate(BUCKETS):
        x = BAR_X0 + i * (BAR_W + BAR_GAP)
        value = state[key]
        h = (value / Y_MAX) * plot_h
        color = COLORS[key]
        if h > 0.5:
            d.rounded_rectangle((x, PLOT_BOTTOM - h, x + BAR_W, PLOT_BOTTOM),
                                radius=8, fill=color)
        if key == active_key and active_t < 1.0:
            pad = 4
            d.rounded_rectangle(
                (x - pad, PLOT_BOTTOM - h - pad, x + BAR_W + pad,
                 PLOT_BOTTOM + pad),
                radius=12, outline=TEXT, width=2)
        shown = int(round(value))
        d.text((x + BAR_W / 2, PLOT_BOTTOM - h - 16), str(shown),
               font=F20 if shown else F16,
               fill=TEXT if shown else MUTED, anchor="mb")
        d.text((x + BAR_W / 2, 563), label, font=F16,
               fill=TEXT if key == active_key else SOFT, anchor="ma")

    # Progress track ------------------------------------------------------
    track_x, track_y, seg_w, seg_gap = 86, 615, 21, 1
    for i, rec in enumerate(records):
        x = track_x + i * (seg_w + seg_gap)
        filled = i < n
        color = COLORS[bucket_of(rec["rank"])] if filled else GRID
        d.rounded_rectangle(
            (x, track_y, x + seg_w, track_y + 11), radius=3, fill=color,
            outline=TEXT if i == active_idx and active_t < 1.0 else None,
            width=2 if i == active_idx and active_t < 1.0 else 1)
        if i % 10 == 0:
            d.text((x + (10 * (seg_w + seg_gap)) / 2, 632),
                   f"G{i // 10 + 1}", font=F14, fill=MUTED, anchor="ma")

    # Ticker --------------------------------------------------------------
    rec = records[active_idx] if records else None
    if rec is not None:
        side = "Blue" if rec["side"] == 0 else "Red"
        rank = rec["rank"]
        rank_text = f"#{rank}" if rank is not None else "not ranked"
        pick_no = sum(1 for j in range(active_idx + 1)
                      if records[j]["game"] == rec["game"])
        label = (f"G{rec['game']} · P{pick_no} · {side} "
                 f"{rec['champion']}  →  {rank_text}")
        detail = f"model #1: {rec['top1']}" if rec["top1"] else ""
        d.rounded_rectangle((64, 662, 1216, 704), radius=12, fill=PANEL)
        d.rectangle((64, 662, 69, 704),
                    fill=COLORS[bucket_of(rank)])
        d.text((86, 683), label, font=F18, fill=TEXT, anchor="lm")
        if detail:
            d.text((1194, 683), detail, font=F16, fill=MUTED, anchor="rm")
    return img


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    records = payload["records"]
    if len(records) != 50:
        raise ValueError(f"expected 50 picks, got {len(records)}")

    states = [counts_after(records, i) for i in range(len(records) + 1)]
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)

    frame_no = 0

    def save(img) -> None:
        nonlocal frame_no
        img.save(FRAMES / f"frame_{frame_no:05d}.png")
        frame_no += 1

    save(draw_frame(states[0], 0, records, 0, 1.0, None))
    for _ in range(INTRO_FRAMES - 1):
        save(draw_frame(states[0], 0, records, 0, 1.0, None))

    for i, rec in enumerate(records):
        key = bucket_of(rec["rank"])
        for step in range(STEP_FRAMES):
            t = (step + 1) / STEP_FRAMES
            state = lerp_states(states[i], states[i + 1], ease(t))
            save(draw_frame(state, i + 1, records, i, t, key))
        if i + 1 < len(records) and records[i + 1]["game"] != rec["game"]:
            for _ in range(GAME_GAP_FRAMES):
                save(draw_frame(states[i + 1], i + 1, records, i, 1.0, None))

    final = draw_frame(states[-1], len(records), records,
                       len(records) - 1, 1.0, None)
    for _ in range(OUTRO_FRAMES):
        save(final)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", str(FRAMES / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUT),
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(FRAMES)
    print(f"wrote {OUT.name}: {frame_no} frames, {frame_no / FPS:.1f}s")


if __name__ == "__main__":
    main()
