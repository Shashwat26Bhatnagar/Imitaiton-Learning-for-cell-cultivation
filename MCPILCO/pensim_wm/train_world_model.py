



"""
Train the PenSim GP world model from an offline buffer, evaluate one-step
prediction quality, and save it for later stages.

Usage:
    python -m pensim_wm.train_world_model \
        --buffer buffers/pensim_offline.pkl \
        --out models/pensim_world_model.pkl \
        --state_stride 10 --reward_stride 10 --n_epoch 800
"""

import argparse

import numpy as np
import torch

from .buffer import TrajectoryBuffer
from .world_model import PenSimWorldModel


def one_step_eval(wm: PenSimWorldModel, buffer: TrajectoryBuffer, stride: int = 25):
    """Held-out-ish one-step diagnostics on buffer transitions (RMSE per channel)."""
    Xo, Xa, Yn, Yr = buffer.stacked_transitions(stride=stride)
    next_pred, _ = wm.predict_next(Xo, Xa)
    rew_pred, _ = wm.predict_reward(Xo, Xa)

    state_rmse = np.sqrt(np.mean((next_pred - Yn) ** 2, axis=0))
    reward_rmse = float(np.sqrt(np.mean((rew_pred - Yr) ** 2)))
    return state_rmse, reward_rmse


def main():
    p = argparse.ArgumentParser(description="Train PenSim GP world model")
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out", type=str, default="models/pensim_world_model.pkl")
    p.add_argument("--state_stride", type=int, default=10)
    p.add_argument("--reward_stride", type=int, default=10)
    p.add_argument("--reward_max_points", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n_epoch", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buf = TrajectoryBuffer.load(args.buffer)
    print(f"[train] loaded buffer: {len(buf)} episodes, {buf.total_transitions} transitions")

    wm = PenSimWorldModel(dtype=torch.float64, device=torch.device("cpu"))
    wm.fit(
        buf,
        state_stride=args.state_stride,
        reward_stride=args.reward_stride,
        reward_max_points=args.reward_max_points,
        state_kwargs={"lr": args.lr, "n_epoch": args.n_epoch},
        reward_kwargs={"lr": max(args.lr, 0.02), "n_epoch": args.n_epoch},
    )

    state_rmse, reward_rmse = one_step_eval(wm, buf)
    print("\n[train] one-step RMSE per obs channel (raw units):")
    for i, v in enumerate(state_rmse):
        print(f"    obs_{i}: {v:.4g}")
    print(f"[train] reward (yield-per-step) RMSE: {reward_rmse:.4g}")

    wm.save(args.out)


if __name__ == "__main__":
    main()
