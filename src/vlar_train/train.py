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
from typing import Optional

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


def train(x_tr, y_tr, m_tr, x_va, y_va, m_va, *, seed: int, cfg: TrainConfig = TrainConfig(),
          ckpt_dir: Optional[Path] = None, w_tr: Optional[np.ndarray] = None,
          w_va: Optional[np.ndarray] = None) -> Trained:
    """`w_tr`, `w_va`: the training and validation rows' sampling weights 1/π_i (amendment 57), in any scale: each
    is normalised to mean 1 over its own rows here. Without them every weight is 1."""
    import torch
    if not len(x_tr) or not len(x_va):
        raise ValueError(f"need training and validation points ({len(x_tr)}, {len(x_va)})")
    w_tr, w_va = unit_mean(w_tr, len(x_tr)), unit_mean(w_va, len(x_va))
    m_tr = m_tr.astype(np.float32) * w_tr.astype(np.float32)[:, None]        # mask × w_i
    m_va = m_va.astype(np.float32) * w_va.astype(np.float32)[:, None]
    floor = float(min(w_tr.min(), w_va.min()))      # Σ mask × w_i of any non-empty batch is at least this
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    norm = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))   # noqa: E731
    xt, yt, mt = norm(x_tr), torch.from_numpy(y_tr.astype(np.float32)), torch.from_numpy(m_tr.astype(np.float32))
    xv, yv, mv = norm(x_va), torch.from_numpy(y_va.astype(np.float32)), torch.from_numpy(m_va.astype(np.float32))
    model = make_model(x_tr.shape[1], cfg.hidden, cfg.dropout)
    if n_params(model) > MAX_PARAMS:
        raise ValueError(f"{n_params(model)} parameters > {MAX_PARAMS} (§8)")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_epoch, since, tr_hist, va_hist = float("inf"), -1, 0, [], []
    best_state = None
    for epoch in range(cfg.epochs):
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
        with torch.no_grad():
            vl = float(_loss(model(xv), yv, mv, floor))
        tr_hist.append(total / max(n, floor))
        va_hist.append(vl)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state": state, "mean": mean, "std": std, "epoch": epoch, "val_loss": vl,
                        "n_in": x_tr.shape[1], "hidden": cfg.hidden}, ckpt_dir / f"epoch{epoch:03d}.pt")
        if vl < best - 1e-9:
            best, best_epoch, best_state, since = vl, epoch, state, 0
        else:
            since += 1
            if since >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if ckpt_dir is not None:
        (ckpt_dir / "best.json").write_text(json.dumps({"best_epoch": best_epoch, "val_loss": va_hist,
                                                        "train_loss": tr_hist, "seed": seed}, indent=1))
    return Trained(model, mean, std, best_epoch, va_hist, tr_hist)
