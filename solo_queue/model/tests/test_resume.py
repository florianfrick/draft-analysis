"""Resume/checkpoint round-trip tests for pretraining/train.py.

Why this file exists: --resume failures are silent. A run that "resumes" but
drops the optimizer moments, the schedule position or the best-so-far counters
still trains and still prints plausible numbers - it just quietly discards
what the earlier epochs learned, or overwrites a better best.pt with a worse
one. So each piece of state is asserted individually.

Loads train.py by path (pretraining/ is not a package) and never touches the
soloQ corpus: a 20-champion toy model is enough to exercise the plumbing.
"""
import importlib.util
import os
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE.parent
TRAIN_PY = MODEL / "pretraining" / "train.py"


def _load_train():
    spec = importlib.util.spec_from_file_location("pretrain_train", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pretrain_train"] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load_train()

SIZE = ["--embed-dim", "32", "--num-intra", "1", "--num-cross", "1",
        "--num-heads", "4"]
N_CHAMPS = 20


def make_args(extra=()):
    """Args for the toy model. parse_args(None) would read sys.argv."""
    return T.parse_args(["--no-wandb"] + SIZE + list(extra))


def make_model(args, num_champs=N_CHAMPS, seed=0):
    torch.manual_seed(seed)
    return T.SiameseBase(
        num_champs=num_champs, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_intra_layers=args.num_intra,
        num_cross_layers=args.num_cross, dropout=args.dropout,
        with_value=not args.no_win, value_seeds=args.value_seeds)


def make_training_state(model, args, total_steps=100):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, T.make_lr_lambda(args, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return opt, sched, scaler


def advance(model, opt, sched, steps=5):
    """Take real optimizer steps so AdamW moments and the LR are non-default."""
    for _ in range(steps):
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.01)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)


def write_ckpt(path, model, args, opt, sched, scaler, step, epoch,
               best_top5, best_win_auc, run_dir=""):
    T.save_ckpt(path, model, args, N_CHAMPS, {"champ_top5": best_top5},
                step, epoch=epoch,
                resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step,
                        "epoch": epoch, "best_top5": best_top5,
                        "best_win_auc": best_win_auc, "run_dir": run_dir,
                        "args": {"epochs": args.epochs}})
    return path


def test_resume_restores_every_piece_of_state(tmp_path):
    args = make_args()
    src = make_model(args, seed=1)
    opt, sched, scaler = make_training_state(src, args)
    advance(src, opt, sched, steps=7)
    assert sched.last_epoch == 7
    lr_before = sched.get_last_lr()[0]

    ckpt = write_ckpt(tmp_path / "latest.pt", src, args, opt, sched, scaler,
                      step=7, epoch=3, best_top5=0.3141, best_win_auc=0.5512,
                      run_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.tmp")), "atomic save left a tmp file"

    dst = make_model(args, seed=999)          # deliberately different weights
    opt2, sched2, scaler2 = make_training_state(dst, args)
    state = T.restore(ckpt, dst, opt2, sched2, scaler2, args,
                      torch.device("cpu"))

    # weights
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), k
    # counters
    assert state == {"epoch": 3, "step": 7, "best_top5": 0.3141,
                     "best_win_auc": 0.5512, "win_stalled": 0,
                     "win_frozen": False, "run_dir": str(tmp_path)}
    # optimizer moments, not just the parameter groups
    assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"]) > 0
    for k, v in opt.state_dict()["state"].items():
        assert torch.equal(v["exp_avg"], opt2.state_dict()["state"][k]["exp_avg"])
        assert torch.equal(v["exp_avg_sq"],
                           opt2.state_dict()["state"][k]["exp_avg_sq"])
    # schedule position: the cosine must continue, not restart at step 0
    assert sched2.last_epoch == 7
    assert sched2.get_last_lr()[0] == pytest.approx(lr_before)


def test_start_epoch_overrides_the_checkpoint_epoch(tmp_path):
    args = make_args()
    model = make_model(args)
    opt, sched, scaler = make_training_state(model, args)
    ckpt = write_ckpt(tmp_path / "latest.pt", model, args, opt, sched, scaler,
                      step=7, epoch=3, best_top5=0.2, best_win_auc=0.5)
    state = T.restore(ckpt, model, opt, sched, scaler,
                      make_args(["--start-epoch", "10"]), torch.device("cpu"))
    assert state["epoch"] == 10
    assert state["step"] == 7                  # rewinding epochs keeps the step


def test_restore_refuses_an_architecture_mismatch(tmp_path):
    args = make_args()
    model = make_model(args)
    opt, sched, scaler = make_training_state(model, args)
    ckpt = write_ckpt(tmp_path / "latest.pt", model, args, opt, sched, scaler,
                      step=1, epoch=1, best_top5=0.0, best_win_auc=0.0)
    wider = make_args(["--embed-dim", "64"])
    big = make_model(wider)
    opt2, sched2, scaler2 = make_training_state(big, wider)
    with pytest.raises(SystemExit) as exc:
        T.restore(ckpt, big, opt2, sched2, scaler2, wider, torch.device("cpu"))
    assert "mismatch" in str(exc.value)


def test_restore_accepts_a_weights_only_checkpoint(tmp_path):
    """best.pt has no training state; resume must degrade, not crash."""
    args = make_args()
    model = make_model(args, seed=3)
    T.save_ckpt(tmp_path / "best.pt", model, args, N_CHAMPS,
                {"champ_top5": 0.5}, 12, epoch=4)
    payload = torch.load(str(tmp_path / "best.pt"), map_location="cpu",
                         weights_only=False)
    assert "resume" not in payload, "best.pt should stay lean"

    dst = make_model(args, seed=4)
    opt, sched, scaler = make_training_state(dst, args)
    state = T.restore(tmp_path / "best.pt", dst, opt, sched, scaler, args,
                      torch.device("cpu"))
    assert state["epoch"] == 4 and state["step"] == 12
    assert state["best_top5"] == -1.0 and state["run_dir"] == ""


def test_resume_reapplies_a_new_lr(tmp_path):
    """--lr must win over the optimizer state, or "resume with a lower lr
    after a divergence" silently keeps the diverging lr."""
    args = make_args()
    model = make_model(args)
    opt, sched, scaler = make_training_state(model, args)
    advance(model, opt, sched, steps=7)
    ckpt = write_ckpt(tmp_path / "latest.pt", model, args, opt, sched, scaler,
                      step=7, epoch=3, best_top5=0.0, best_win_auc=0.0)
    # still on the old peak (a few steps into warmup, so not exactly 1e-3)
    assert opt.param_groups[0]["lr"] > 9e-4

    lower = make_args(["--lr", "3e-4", "--weight-decay", "0.05"])
    dst = make_model(lower, seed=5)
    opt2, sched2, scaler2 = make_training_state(dst, lower)
    T.restore(ckpt, dst, opt2, sched2, scaler2, lower, torch.device("cpu"))

    assert opt2.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert opt2.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert sched2.base_lrs == [pytest.approx(3e-4)]
    sched2.step()                     # the cosine must be rebased on 3e-4
    assert sched2.get_last_lr()[0] <= 3e-4 + 1e-12


def test_win_weight_ramps_then_freezes_on_stall():
    """lambda_win: linear ramp, then hard 0 once the head stops generalising."""
    args = make_args()                      # win_warmup 5, patience 8
    assert args.win_warmup_epochs == 5.0 and args.win_patience == 8
    assert T.win_weight(args, 1, 0) == pytest.approx(0.2)
    assert T.win_weight(args, 3, 0) == pytest.approx(0.6)
    assert T.win_weight(args, 5, 0) == pytest.approx(1.0)
    assert T.win_weight(args, 40, 0) == pytest.approx(1.0)
    # stalled short of patience: still full weight
    assert T.win_weight(args, 20, 7) == pytest.approx(1.0)
    # at patience: cut, and stay cut regardless of epoch
    assert T.win_weight(args, 20, 8) == 0.0
    assert T.win_weight(args, 40, 12) == 0.0
    # explicit opt-outs
    assert T.win_weight(make_args(["--win-patience", "0"]), 40, 99) == \
        pytest.approx(1.0)
    assert T.win_weight(make_args(["--no-win"]), 3, 0) == 0.0
    assert T.win_weight(make_args(["--win-warmup-epochs", "0"]), 1, 0) == \
        pytest.approx(1.0)


def test_win_freeze_latches_even_if_the_stall_counter_resets():
    """Regression: the freeze used to switch back on. With the loss frozen the
    head stops moving but the encoder keeps improving, so the frozen head's AUC
    drifts up ~1e-4, resets the stall counter, and training resumes - observed
    at epoch 26 of run v2-D512-p8 (AUC then fell 0.5597 -> 0.5467)."""
    args = make_args()
    # a reset counter (stalled 0) must NOT unfreeze once latched
    assert T.win_weight(args, 30, 0, frozen=True) == 0.0
    assert T.win_weight(args, 30, 99, frozen=True) == 0.0
    # not latched: normal behaviour
    assert T.win_weight(args, 30, 0, frozen=False) == pytest.approx(1.0)
    # --win-freeze is the explicit manual override
    assert T.win_weight(make_args(["--win-freeze"]), 1, 0, frozen=True) == 0.0


def test_win_freeze_flag_and_persistence(tmp_path):
    """--win-freeze must survive a resume via the stored state, otherwise a
    restarted run silently resumes damaging the encoder."""
    args = make_args(["--win-freeze"])
    assert args.win_freeze is True
    assert make_args().win_freeze is False

    model = make_model(args)
    opt, sched, scaler = make_training_state(model, args)
    path = tmp_path / "latest.pt"
    T.save_ckpt(path, model, args, N_CHAMPS, {"champ_top5": 0.2}, 7, epoch=3,
                resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": 7, "epoch": 3,
                        "best_top5": 0.2, "best_win_auc": 0.56,
                        "win_stalled": 0, "win_frozen": True,
                        "run_dir": str(tmp_path),
                        "args": {"epochs": args.epochs}})
    dst = make_model(args, seed=7)
    opt2, sched2, scaler2 = make_training_state(dst, args)
    plain = make_args()                       # no --win-freeze this time
    state = T.restore(path, dst, opt2, sched2, scaler2, plain,
                      torch.device("cpu"))
    assert state["win_frozen"] is True
    assert T.win_weight(plain, 10, 0, state["win_frozen"]) == 0.0


def test_resume_restores_the_win_stall_counter(tmp_path):
    """A resumed run must not restart the patience count, or an overfitting
    value head gets a fresh 8 epochs of damage on every restart."""
    args = make_args()
    model = make_model(args)
    opt, sched, scaler = make_training_state(model, args)
    path = tmp_path / "latest.pt"
    T.save_ckpt(path, model, args, N_CHAMPS, {"champ_top5": 0.2}, 7, epoch=3,
                resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": 7, "epoch": 3,
                        "best_top5": 0.2, "best_win_auc": 0.56,
                        "win_stalled": 6, "run_dir": str(tmp_path),
                        "args": {"epochs": args.epochs}})
    dst = make_model(args, seed=6)
    opt2, sched2, scaler2 = make_training_state(dst, args)
    state = T.restore(path, dst, opt2, sched2, scaler2, args,
                      torch.device("cpu"))
    assert state["win_stalled"] == 6
    assert T.win_weight(args, 10, state["win_stalled"]) == pytest.approx(1.0)
    assert T.win_weight(args, 10, state["win_stalled"] + 2) == 0.0


def test_find_resume_resolves_run_names_under_the_ckpt_root(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(T, "CKPT_ROOT", tmp_path)
    run = tmp_path / "run_20260101_000000_demo"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"x")
    # short name, full relative path, directory, and 'auto' all resolve
    assert T.find_resume_ckpt("run_20260101_000000_demo") == run / "latest.pt"
    assert T.find_resume_ckpt(str(run)) == run / "latest.pt"
    assert T.find_resume_ckpt("auto") == run / "latest.pt"
    with pytest.raises(SystemExit):
        T.find_resume_ckpt("run_does_not_exist")


def test_find_resume_auto_takes_the_newest_run(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CKPT_ROOT", tmp_path)
    old, new = tmp_path / "run_a", tmp_path / "run_b"
    for d, mtime in ((old, 1000), (new, 2000)):
        d.mkdir()
        (d / "latest.pt").write_bytes(b"x")
        os.utime(d / "latest.pt", (mtime, mtime))
    assert T.find_resume_ckpt("auto") == new / "latest.pt"


def test_limit_drafts_subsamples_the_cache_instead_of_scanning_a_prefix(
        tmp_path, monkeypatch):
    """Regression: the prefix scan kept 0 drafts because the corpus opens with
    the previous patch, which then divided by zero on the base-rate print."""
    import types
    monkeypatch.setattr(T, "P", types.SimpleNamespace(BASE_DIR=tmp_path))
    rows = [((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12))] * 500
    torch.save({"schema": 2, "patch": "16.20", "rows": rows,
                "wins": [1] * 500}, str(tmp_path / "drafts_16.20.pt"))
    monkeypatch.setattr(T, "load_mapping", lambda patch: ({}, 173))
    args = make_args(["--limit-drafts", "50"])
    got_rows, got_wins, _ = T.load_rows(args)
    assert len(got_rows) == 50 and len(got_wins) == 50
    # evenly strided across the whole cache, so every region is represented
    assert got_rows[0] == rows[0]
    assert got_rows[1] == rows[10]
