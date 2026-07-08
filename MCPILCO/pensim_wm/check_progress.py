#!/usr/bin/env python3
"""
Summarize a pretrain_bcnp_synthetic progress log (JSONL) so you can check on a
long/unattended Slurm job at a glance.

Usage:
    poetry run python pensim_wm/slurm/check_progress.py pretrained/bcnp_24var.pt.progress.jsonl
    watch -n 30 poetry run python pensim_wm/slurm/check_progress.py pretrained/bcnp_24var.pt.progress.jsonl
"""
import json
import sys


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"(empty) {path}")
        return

    start = next((r for r in records if r["event"] == "start"), None)
    epochs = [r for r in records if r["event"] == "epoch"]
    evals = [r for r in records if r["event"] == "eval"]
    ckpts = [r for r in records if r["event"] == "checkpoint"]
    done = any(r["event"] == "done" for r in records)

    print(f"=== progress: {path} ===")
    if start:
        print(f"  target epochs: {start.get('epochs')}  num_nodes: {start.get('num_nodes')}")
    if epochs:
        last = epochs[-1]
        print(f"  latest epoch:  {last['epoch']}/{last.get('epochs','?')}  loss={last['loss']:.4f}")
        recent = epochs[-5:]
        print(f"  loss (last {len(recent)}): " + ", ".join(f"{r['loss']:.3f}" for r in recent))
    if evals:
        print(f"  held-out AUC over time:")
        for r in evals:
            flag = "  <-- best" if r["held_out_auc"] == max(e["held_out_auc"] for e in evals) else ""
            print(f"    epoch {r['epoch']:>4}: AUC={r['held_out_auc']:.3f}  "
                  f"mean_edge_prob={r['mean_edge_prob']:.3f}{flag}")
    if ckpts:
        print(f"  last checkpoint: epoch {ckpts[-1]['epoch']} -> {ckpts[-1]['path']}")
    print(f"  status: {'DONE' if done else 'IN PROGRESS (or killed before finishing)'}")


if __name__ == "__main__":
    main()
