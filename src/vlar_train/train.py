"""Training one decider in one fold (SPEC §8): from scratch (a fresh initialisation seeded by fold and decider, no
weights carried over), on the training tasks' points; a checkpoint every epoch; the stopping epoch is the one with the
lowest validation loss (early stopping with patience). Loss: binary cross-entropy of each branch's predicted success
probability against its label (the fraction of seeds that succeeded), over the branches the point has labels for.
Inputs are standardised with the training points' mean and standard deviation (kept with the checkpoint).

Sampling weights (amendment 57 (4)): the loss is each head's BCE × mask × w_i, divided by Σ mask × w_i, with
w_i = 1/π_i normalised to mean 1 within the rows it is used on; early stopping uses the same weighted loss on the
validation task. With every weight 1 this is the unweighted training, bit for bit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .deciders import make_model, n_params

MAX_PARAMS = 5_000_000


@dataclass
class TrainConfig:
    epochs: int = 60
    patience: int = 8
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: tuple[int, ...] = (512, 256)
    dropout: float = 0.1
    seeds: int = 5              # training seeds per decider; their predicted probabilities are averaged (`train_seeds`)


@dataclass
class Trained:
    model: object
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    val_loss: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            z = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
            return torch.sigmoid(self.model(z)).numpy().astype(np.float64)


def weighted_bce(p: np.ndarray, y: np.ndarray, m: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """The training loss of given probabilities, in NumPy: Σ BCE × mask × w_i / Σ mask × w_i (amendment 57 (4)),
    w_i normalised to mean 1 over the rows given."""
    mw = np.asarray(m, dtype=np.float64) * unit_mean(w, len(y))[:, None]
    p = np.clip(np.broadcast_to(np.asarray(p, dtype=np.float64), y.shape), 1e-7, 1 - 1e-7)
    total = float(mw.sum())
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p)) * mw).sum() / total) if total > 0 else 0.0


def constant_predictor(y_tr: np.ndarray, m_tr: np.ndarray, w_tr: Optional[np.ndarray] = None) -> np.ndarray:
    """The baseline that uses no input: per head, the weighted mean of the training rows' labels,
    Σ w_i × mask × label / Σ w_i × mask (0.5 for a head with no label, which the mask keeps out of every loss).
    It is the constant that minimises the weighted training loss, so a decider whose validation loss is not below
    this predictor's has learned nothing that transfers to the validation task."""
    mw = np.asarray(m_tr, dtype=np.float64) * unit_mean(w_tr, len(y_tr))[:, None]
    den = mw.sum(axis=0)
    return np.where(den > 0, (mw * y_tr).sum(axis=0) / np.where(den > 0, den, 1.0), 0.5)


def seed_for(fold: int, decider: str, base: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{fold}|{decider}".encode()).digest()[:4], "little")


def _loss(logits, y, m, w):
    """Σ BCE × mask × w_i / Σ mask × w_i over the rows and heads given (amendment 57 (4)); `m` already holds
    mask × w_i (`weighted_mask`), so with every w_i = 1 it is the mask and the loss the unweighted one."""
    import torch.nn.functional as F
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m
    return per.sum() / m.sum().clamp(min=w)


def unit_mean(w: Optional[np.ndarray], n: int) -> np.ndarray:
    """w_i normalised to mean 1 over the rows given (amendment 57 (3)); all 1 when no weights are given."""
    if w is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,) or not (np.isfinite(w) & (w > 0)).all():
        raise ValueError("weights must be one finite positive number per row")
    return w / w.mean()


def train(x_tr, y_tr, m_tr, x_va=None, y_va=None, m_va=None, *, seed: int, cfg: TrainConfig = TrainConfig(),
          ckpt_dir: Optional[Path] = None, w_tr: Optional[np.ndarray] = None,
          w_va: Optional[np.ndarray] = None, stop_epoch: Optional[int] = None) -> Trained:
    """`w_tr`, `w_va`: the training and validation rows' sampling weights 1/π_i (amendment 57), in any scale: each
    is normalised to mean 1 over its own rows here. Without them every weight is 1.

    `stop_epoch`: train to that epoch (counted from 0, as `best_epoch` is) and keep the model as it then is: no
    validation rows, no early stopping (§9's collectors, trained on every task). Without it, early stopping on the
    validation rows as §8 says."""
    import torch
    fixed = stop_epoch is not None
    if fixed and int(stop_epoch) < 0:
        raise ValueError(f"stop_epoch counts from 0, not {stop_epoch}")
    if not len(x_tr) or (not fixed and (x_va is None or not len(x_va))):
        raise ValueError(f"need training and validation points ({len(x_tr)}, {0 if x_va is None else len(x_va)})")
    w_tr = unit_mean(w_tr, len(x_tr))
    m_tr = m_tr.astype(np.float32) * w_tr.astype(np.float32)[:, None]        # mask × w_i
    if fixed:
        floor = float(w_tr.min())
    else:
        w_va = unit_mean(w_va, len(x_va))
        m_va = m_va.astype(np.float32) * w_va.astype(np.float32)[:, None]
        floor = float(min(w_tr.min(), w_va.min()))  # Σ mask × w_i of any non-empty batch is at least this
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    norm = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))   # noqa: E731
    xt, yt, mt = norm(x_tr), torch.from_numpy(y_tr.astype(np.float32)), torch.from_numpy(m_tr.astype(np.float32))
    if not fixed:
        xv, yv, mv = norm(x_va), torch.from_numpy(y_va.astype(np.float32)), torch.from_numpy(m_va.astype(np.float32))
    model = make_model(x_tr.shape[1], cfg.hidden, cfg.dropout)
    if n_params(model) > MAX_PARAMS:
        raise ValueError(f"{n_params(model)} parameters > {MAX_PARAMS} (§8)")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_epoch, since, tr_hist, va_hist = float("inf"), -1, 0, [], []
    best_state = None
    for epoch in range(int(stop_epoch) + 1 if fixed else cfg.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=g)
        total, n = 0.0, 0
        for k in range(0, len(order), cfg.batch):
            idx = order[k:k + cfg.batch]
            loss = _loss(model(xt[idx]), yt[idx], mt[idx], floor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bw = float(mt[idx].sum())                        # the batch's Σ mask × w_i
            total, n = total + float(loss.detach()) * bw, n + bw
        model.eval()
        if fixed:
            vl = None
        else:
            with torch.no_grad():
                vl = float(_loss(model(xv), yv, mv, floor))
        tr_hist.append(total / max(n, floor))
        va_hist.append(vl)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state": state, "mean": mean, "std": std, "epoch": epoch, "val_loss": vl,
                        "n_in": x_tr.shape[1], "hidden": cfg.hidden}, ckpt_dir / f"epoch{epoch:03d}.pt")
        if fixed:
            best_epoch, best_state = epoch, state
        elif vl < best - 1e-9:
            best, best_epoch, best_state, since = vl, epoch, state, 0
        else:
            since += 1
            if since >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if ckpt_dir is not None:
        (ckpt_dir / "best.json").write_text(json.dumps({"best_epoch": best_epoch, "val_loss": va_hist,
                                                        "train_loss": tr_hist, "seed": seed,
                                                        **({"stop_epoch": int(stop_epoch)} if fixed else {})},
                                                       indent=1))
    return Trained(model, mean, std, best_epoch, va_hist, tr_hist)


@dataclass
class Averaged:
    """A decider as the mean of its training seeds: each member is trained from scratch with its own seed and
    stopped early on its own validation loss; the decider's probabilities are the mean of the members' (so its
    score p_one_call - p_continue is the mean of theirs). With one member it is that member, bit for bit."""
    members: list
    seeds: list

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(self.members) == 1:
            return self.members[0].predict(x)
        return np.mean([m.predict(x) for m in self.members], axis=0)

    @property
    def best_epochs(self) -> list:
        return [int(m.best_epoch) for m in self.members]

    @property
    def best_epoch(self):
        """The member's stopping epoch, or with several the median of theirs (it can be fractional)."""
        return self.members[0].best_epoch if len(self.members) == 1 else float(np.median(self.best_epochs))

    @property
    def member_val_loss(self) -> list:
        return [float(m.val_loss[m.best_epoch]) for m in self.members]

    @property
    def epochs_run(self) -> list:
        return [len(m.val_loss) for m in self.members]

    def val_loss(self, x_va, y_va, m_va, w_va=None) -> float:
        """The validation loss of the averaged prediction; with one member, that member's own at its best epoch."""
        if len(self.members) == 1:
            return self.members[0].val_loss[self.members[0].best_epoch]
        return weighted_bce(self.predict(x_va), y_va, m_va, w_va)


def member_seeds(fold: int, name: str, n: int, base: int = 0) -> list[int]:
    """The training seeds of a decider's members: the first is the seed a single run always had."""
    if n < 1:
        raise ValueError(f"at least one training seed, not {n}")
    return [seed_for(fold, name, base)] + [seed_for(fold, f"{name}|seed{k}", base) for k in range(1, n)]


def train_seeds(x_tr, y_tr, m_tr, x_va=None, y_va=None, m_va=None, *, seeds: Sequence[int],
                cfg: TrainConfig = TrainConfig(), ckpt_dir: Optional[Path] = None, w_tr: Optional[np.ndarray] = None,
                w_va: Optional[np.ndarray] = None, stop_epoch: Optional[int] = None) -> Averaged:
    """`train` once per seed, every member on the same rows and settings. Checkpoints: with one seed in `ckpt_dir`
    as before, with several in `ckpt_dir/seed<k>`."""
    seeds = [int(x) for x in seeds]
    members = [train(x_tr, y_tr, m_tr, x_va, y_va, m_va, seed=sd, cfg=cfg, w_tr=w_tr, w_va=w_va, stop_epoch=stop_epoch,
                     ckpt_dir=None if ckpt_dir is None else (ckpt_dir if len(seeds) == 1 else ckpt_dir / f"seed{k}"))
               for k, sd in enumerate(seeds)]
    return Averaged(members, seeds)
