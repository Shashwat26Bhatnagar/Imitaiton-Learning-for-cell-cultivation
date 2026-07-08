"""
FILE 2 of the causal-discovery stage.

Imports the transformer-based Bayesian Causal Neural Process (BCNP,
``CausalProbabilisticDecoder`` from CausalStructureNeuralProcess) and uses it to
predict the posterior over causal masks for the datasets produced by file 1
(build_causal_datasets.py).

Two modes
---------
* ``infer``  : load a (optionally pretrained) BCNP, feed the file-1 datasets, and
               output the posterior over causal masks -- both a marginal
               edge-probability matrix (D x D) and sampled DAGs. Needs NO labels.

* ``train``  : supervised meta-training of the BCNP on file-1 datasets. This only
               makes sense if the datasets carry a ground-truth DAG label
               (``has_labels=True`` in the HDF5, i.e. you built them with
               ``--assumed_adjacency``). Real PenSim trajectories have no
               ground-truth graph, so the standard recipe is: pretrain on
               synthetic labelled data (the BCNP repo's own generator), then use
               ``infer`` here on PenSim. See the note printed by ``train``.

Locate the BCNP repo via the ``CSNP_ROOT`` env var, or ``--csnp_root``, or a
sibling ``BayesianDAG/`` (or ``CausalStructureNeuralProcess/``) folder.

Cluster deps (Python 3.10): the BCNP package needs ``attrdict3`` (NOT ``attrdict``,
which imports ``collections.Mapping`` and breaks on 3.10+), plus ``dill h5py
tqdm einops``.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# locate + import the BCNP repo
# ---------------------------------------------------------------------------
def _add_csnp_to_path(csnp_root=None):
    _root_up = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        csnp_root,
        os.environ.get("CSNP_ROOT"),
        os.path.join(os.getcwd(), "BayesianDAG"),
        os.path.join(os.getcwd(), "CausalStructureNeuralProcess"),
        os.path.join(_root_up, "BayesianDAG"),
        os.path.join(_root_up, "CausalStructureNeuralProcess"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "ml2_meta_causal_discovery")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise FileNotFoundError(
        "Could not find the BCNP repo (BayesianDAG / CausalStructureNeuralProcess). "
        f"Set CSNP_ROOT or pass --csnp_root. Tried: {[c for c in candidates if c]}"
    )


def build_model(num_nodes, d_model=128, dim_feedforward=256, nhead=8,
                num_layers_encoder=4, num_layers_decoder=4, n_perm_samples=100,
                sinkhorn_iter=1000, use_positional_encoding=False,
                device="cpu", dtype=torch.float32):
    """Instantiate the BCNP probabilistic decoder (encoder/decoder layers must be even)."""
    from ml2_meta_causal_discovery.models.causaltransformernp import CausalProbabilisticDecoder
    assert num_layers_encoder % 2 == 0, "num_layers_encoder must be even"
    model = CausalProbabilisticDecoder(
        d_model=d_model, emb_depth=1, dim_feedforward=dim_feedforward, nhead=nhead,
        dropout=0.0, num_layers_encoder=num_layers_encoder,
        num_layers_decoder=num_layers_decoder, num_nodes=num_nodes,
        n_perm_samples=n_perm_samples, sinkhorn_iter=sinkhorn_iter,
        use_positional_encoding=use_positional_encoding, Q_before_L=False,
        device=device, dtype=dtype,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# HDF5 helpers (read file-1 output directly, incl. attrs)
# ---------------------------------------------------------------------------
def _read_hdf5(path):
    with h5py.File(path, "r") as f:
        data = f["data"][:]                       # (B, N, D)
        label = f["label"][:]                     # (B, D, D)
        has_labels = bool(f.attrs.get("has_labels", False))
        node_names = f.attrs.get("node_names", None)
        if node_names is not None:
            node_names = [n.decode() if isinstance(n, bytes) else str(n) for n in node_names]
    return data, label, has_labels, node_names


def _normalize_samples(x):
    """Normalize across the sample axis (dim=1), matching BCNP training."""
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True)
    return (x - mu) / (sd + 1e-8)


# ---------------------------------------------------------------------------
# INFERENCE: posterior over causal masks
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_posterior(model, hdf5_paths, num_samples=100, batch_size=16,
                    device="cpu", dtype=torch.float32):
    """Return (marginal_edge_prob (D,D), all_dag_samples (S_total,D,D), node_names)."""
    model.eval()
    marg_accum = []
    dag_samples = []
    node_names = None
    for path in hdf5_paths:
        data, _label, _has, names = _read_hdf5(path)
        node_names = names if names is not None else node_names
        B = data.shape[0]
        for i in range(0, B, batch_size):
            x = torch.tensor(np.nan_to_num(data[i:i + batch_size]), dtype=dtype, device=device)
            x = _normalize_samples(x)
            probs = model.forward(x, graph=None, is_training=False, mask=None)  # (S,b,D,D)
            marg_accum.append(probs.mean(0).cpu())                              # (b,D,D)
            samples, _ = model.sample(target_data=x, num_samples=num_samples, mask=None)
            dag_samples.append(samples.reshape(-1, samples.shape[-2], samples.shape[-1]).cpu())
    marginal = torch.cat(marg_accum, 0).mean(0).numpy()      # (D,D) posterior marginal edge probs
    all_dags = torch.cat(dag_samples, 0).numpy()             # (S_total, D, D)
    return marginal, all_dags, node_names


def save_marginal(marginal, node_names, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(out_prefix + "_edge_prob.npy", marginal)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        D = marginal.shape[0]
        plt.figure(figsize=(7, 6))
        im = plt.imshow(marginal, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="P(edge i -> j)")
        if node_names and len(node_names) == D:
            plt.xticks(range(D), node_names, rotation=90, fontsize=6)
            plt.yticks(range(D), node_names, fontsize=6)
        plt.xlabel("target (child)"); plt.ylabel("source (parent)")
        plt.title("BCNP posterior: marginal edge probabilities")
        plt.tight_layout(); plt.savefig(out_prefix + "_edge_prob.png", dpi=150); plt.close()
    except Exception as e:
        print(f"[infer] (skipped heatmap: {e})")


# ---------------------------------------------------------------------------
# TRAINING: supervised meta-training (requires labels)
# ---------------------------------------------------------------------------
def train_supervised(model, hdf5_paths, epochs=10, lr=1e-4, batch_size=16,
                     device="cpu", dtype=torch.float32, save_path=None,
                     eval_files=None, eval_every=5, checkpoint_every=5,
                     progress_log=None, resume_path=None, resume=True):
    """
    Supervised meta-training with progress markers AND resume for long/unattended
    or interruptible runs (e.g. login-node runs that may get killed):
      - every `checkpoint_every` epochs: save (a) plain model weights to `save_path`
        (for `infer`), and (b) a full resume state {model, optimizer, epoch} to
        `resume_path` (default `<save_path>.resume`).
      - on startup, if `resume` and `resume_path` exists, reload model+optimizer
        and CONTINUE from the next epoch instead of restarting.
      - every `eval_every` epochs (if eval_files): held-out AUC -> progress log.
    """
    # guard: need labels
    for path in hdf5_paths:
        _, _, has_labels, _ = _read_hdf5(path)
        if not has_labels:
            raise ValueError(
                f"{path} has no DAG labels (has_labels=False). Supervised training "
                "needs a ground-truth graph per dataset. Either (a) rebuild file-1 "
                "datasets with --assumed_adjacency, or (b) pretrain on synthetic "
                "labelled data (BCNP repo generator) and use `infer` here instead."
            )
    from ml2_meta_causal_discovery.utils.datautils import MultipleFileDataset
    ds = MultipleFileDataset(list(hdf5_paths))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if resume_path is None and save_path is not None:
        resume_path = save_path + ".resume"
    if progress_log:
        os.makedirs(os.path.dirname(progress_log) or ".", exist_ok=True)

    def _log(record):
        line = json.dumps(record)
        print(f"[progress] {line}", flush=True)
        if progress_log:
            with open(progress_log, "a") as f:
                f.write(line + "\n")

    # --- resume ------------------------------------------------------------
    start_epoch = 0
    if resume and resume_path and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        try:
            opt.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"[train] warning: could not restore optimizer state ({e}); continuing.")
        start_epoch = int(state.get("epoch", 0))
        print(f"[train] RESUMING from {resume_path} at epoch {start_epoch + 1}/{epochs}")
        _log({"event": "resume", "from_epoch": start_epoch})
        if start_epoch >= epochs:
            print(f"[train] already trained {start_epoch} >= {epochs} epochs; nothing to do.")
            return model

    def _save_resume(epoch_done):
        if resume_path:
            os.makedirs(os.path.dirname(resume_path) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch_done}, resume_path)

    _log({"event": "start", "epochs": epochs, "start_epoch": start_epoch,
          "n_files": len(hdf5_paths),
          "num_nodes": model.num_nodes if hasattr(model, "num_nodes") else None})

    model.train()
    for epoch in range(start_epoch, epochs):
        total, nb = 0.0, 0
        for data, graph in loader:
            x = _normalize_samples(torch.nan_to_num(data.to(device, dtype=dtype)))
            g = graph.to(device, dtype=dtype)
            opt.zero_grad()
            logits = model(x, graph=g, mask=None)
            loss = model.calculate_loss(logits, g).mean()
            loss.backward()
            opt.step()
            total += float(loss.item()); nb += 1
        avg_loss = total / max(nb, 1)
        print(f"[train] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        _log({"event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": avg_loss})

        do_ckpt = checkpoint_every and (epoch + 1) % checkpoint_every == 0
        do_eval = eval_files and eval_every and (epoch + 1) % eval_every == 0

        if do_ckpt:
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_path)   # plain weights for infer
            _save_resume(epoch + 1)                          # full resume state
            _log({"event": "checkpoint", "epoch": epoch + 1, "path": save_path})

        if do_eval:
            from sklearn.metrics import roc_auc_score
            model.eval()
            with torch.no_grad():
                probs_all, labels_all = [], []
                for path in eval_files:
                    edata, elabel, _has, _names = _read_hdf5(path)
                    D = edata.shape[-1]
                    offdiag = ~np.eye(D, dtype=bool)
                    for i in range(0, edata.shape[0], batch_size):
                        ex = torch.tensor(np.nan_to_num(edata[i:i + batch_size]),
                                          dtype=dtype, device=device)
                        ex = _normalize_samples(ex)
                        p = model.forward(ex, graph=None, is_training=False, mask=None).mean(0)
                        p = p.cpu().numpy()
                        lab = elabel[i:i + batch_size]
                        for b in range(p.shape[0]):
                            probs_all.append(p[b][offdiag]); labels_all.append(lab[b][offdiag])
            model.train()
            probs_all = np.concatenate(probs_all); labels_all = np.concatenate(labels_all)
            auc = float(roc_auc_score(labels_all, probs_all)) if labels_all.max() != labels_all.min() else float("nan")
            _log({"event": "eval", "epoch": epoch + 1, "held_out_auc": auc,
                  "mean_edge_prob": float(probs_all.mean())})

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[train] saved checkpoint -> {save_path}")
    _save_resume(epochs)
    _log({"event": "final_checkpoint", "path": save_path})
    _log({"event": "done"})
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _common_model_args(p):
    p.add_argument("--csnp_root", type=str, default=None)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers_encoder", type=int, default=4)
    p.add_argument("--num_layers_decoder", type=int, default=4)
    p.add_argument("--n_perm_samples", type=int, default=100)
    p.add_argument("--sinkhorn_iter", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser(description="BCNP: posterior over causal masks")
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("infer", help="predict posterior over causal masks (no labels needed)")
    pi.add_argument("--data", nargs="+", required=True, help="file-1 HDF5 path(s)")
    pi.add_argument("--checkpoint", type=str, default=None, help="pretrained BCNP weights (.pt)")
    pi.add_argument("--num_samples", type=int, default=100)
    pi.add_argument("--out_prefix", type=str, default="causal_out/pensim_posterior")
    _common_model_args(pi)

    pt = sub.add_parser("train", help="supervised meta-training (needs labelled datasets)")
    pt.add_argument("--data", nargs="+", required=True, help="file-1 HDF5 path(s) with labels")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--batch_size", type=int, default=16)
    pt.add_argument("--save_path", type=str, default="causal_out/bcnp.pt")
    pt.add_argument("--eval_data", nargs="*", default=None, help="held-out HDF5 for progress AUC")
    pt.add_argument("--eval_every", type=int, default=5)
    pt.add_argument("--checkpoint_every", type=int, default=5)
    pt.add_argument("--progress_log", type=str, default=None)
    _common_model_args(pt)

    args = p.parse_args()
    _add_csnp_to_path(args.csnp_root)

    # infer num_nodes from the first data file
    D = _read_hdf5(args.data[0])[0].shape[-1]
    model = build_model(
        num_nodes=D, d_model=args.d_model, dim_feedforward=args.dim_feedforward,
        nhead=args.nhead, num_layers_encoder=args.num_layers_encoder,
        num_layers_decoder=args.num_layers_decoder, n_perm_samples=args.n_perm_samples,
        sinkhorn_iter=args.sinkhorn_iter, device=args.device,
    )
    print(f"[bcnp] model with num_nodes={D} on {args.device}")

    if args.mode == "infer":
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
            print(f"[bcnp] loaded checkpoint {args.checkpoint}")
        else:
            print("[bcnp] WARNING: no --checkpoint given; using randomly-initialised weights. "
                  "For meaningful posteriors, load a BCNP pretrained on synthetic data.")
        marginal, dags, names = infer_posterior(
            model, args.data, num_samples=args.num_samples, device=args.device)
        save_marginal(marginal, names, args.out_prefix)
        print(f"[bcnp] posterior over causal masks:")
        print(f"       marginal edge-prob matrix {marginal.shape} -> {args.out_prefix}_edge_prob.npy")
        print(f"       {dags.shape[0]} sampled DAGs; mean edges/graph = {dags.sum((1,2)).mean():.2f}")
    elif args.mode == "train":
        train_supervised(
            model, args.data, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device, save_path=args.save_path,
            eval_files=args.eval_data, eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every, progress_log=args.progress_log)


if __name__ == "__main__":
    main()
