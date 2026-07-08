"""
Pretrain the BCNP (CausalProbabilisticDecoder) on SYNTHETIC labelled data, so it
can then be loaded by ``bcnp_causal.py infer`` to give a real posterior over
causal masks on PenSim.

Why synthetic: BCNP is amortized Bayesian inference -- it learns the *procedure*
`dataset -> P(graph | dataset)` from many (dataset, known-DAG) pairs. Real PenSim
has no ground-truth DAG, so labels only exist in synthetic data (where we build
the SCM ourselves). See the ICLR paper's 2-variable sanity check.

Option B (no tensorflow): the repo's `gp` function generator needs gpflow/TF, but
`gplvm` (torch GP-latent-variable), `neuralnet` (torch), and `linear` (numpy) do
not -- they only fail because `functions_generator.py` imports gpflow/TF at module
top level. We install a meta-path shim that fabricates dummy gpflow/tensorflow
modules, so those three torch/numpy generators import and run with no TF present.

Pipeline: generate labelled HDF5 (data,label) -> train CausalProbabilisticDecoder
(reusing bcnp_causal.train_supervised) -> save checkpoint (+ meta.json). The
checkpoint's num_nodes MUST match the node count you later infer on (15 for the
PenSim `sa` set); the architecture is fixed to num_nodes, so pretrain at the same D.

Start small: a 2-variable run verifies the model recovers a known posterior
(edge-direction AUC well above 0.5) before committing to a long 15-node run.
"""

import argparse
import glob
import importlib.abc
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch


# ===========================================================================
# tensorflow / gpflow shim (lets torch/numpy generators import without TF)
# ===========================================================================
class _AnyType:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _AnyType()
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _AnyType
    def __getitem__(self, k): return _AnyType


def _make_fake(name):
    m = types.ModuleType(name)
    m.__file__ = f"<shim:{name}>"
    m.__path__ = []          # package -> allows submodule imports
    m.__spec__ = None
    def _getattr(k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _AnyType
    m.__getattr__ = _getattr
    return m


class _ShimLoader(importlib.abc.Loader):
    def create_module(self, spec): return _make_fake(spec.name)
    def exec_module(self, module): pass


class _ShimFinder(importlib.abc.MetaPathFinder):
    PREFIXES = ("gpflow", "tensorflow", "tensorflow_probability")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.PREFIXES:
            return importlib.util.spec_from_loader(name, _ShimLoader())
        return None


def install_tf_shims():
    """Fabricate gpflow/tensorflow modules so gplvm/neuralnet/linear generators import."""
    if not any(isinstance(f, _ShimFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ShimFinder())
    # attrdict on py>=3.10 needs collections.Mapping etc.
    import collections, collections.abc
    for n in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))


# ===========================================================================
# synthetic data generation (wraps the repo's ClassifyDatasetGenerator)
# ===========================================================================
def _graph_degrees(num_vars, exp_edges_lower, exp_edges_upper):
    lo = int(round(exp_edges_lower * num_vars))
    hi = int(round(exp_edges_upper * num_vars))
    degs = list(range(max(lo, 0), max(hi, lo) + 1))
    return degs or [0]


def generate_synthetic(num_vars, out_dir, generators=("gplvm", "neuralnet", "linear"),
                       n_files=6, datasets_per_file=64, num_samples=400,
                       graph_types=("ER",), exp_edges_lower=0.0, exp_edges_upper=1.0,
                       seed=0, verbose=True):
    """Generate labelled HDF5 files (data,label,has_labels=True). Returns file paths."""
    install_tf_shims()
    import h5py
    from ml2_meta_causal_discovery.datasets.dataset_generators import ClassifyDatasetGenerator

    os.makedirs(out_dir, exist_ok=True)
    degs = _graph_degrees(num_vars, exp_edges_lower, exp_edges_upper)
    paths = []
    for fi in range(n_files):
        fg = generators[fi % len(generators)]         # cycle through the mixture
        path = os.path.join(out_dir, f"synth_{num_vars}var_{fg}_{fi}.hdf5")
        if os.path.exists(path):                       # resume-friendly: skip existing
            paths.append(path)
            if verbose:
                print(f"[gen] {os.path.basename(path)}: exists, skipping")
            continue
        np.random.seed(seed + fi)
        torch.manual_seed(seed + fi)
        gen = ClassifyDatasetGenerator(
            num_variables=num_vars, function_generator=fg,
            batch_size=datasets_per_file, num_samples=num_samples,
            graph_type=list(graph_types), graph_degrees=degs,
            kernel_sum=True, mean_function="latent",
        )
        data, graphs = next(gen.generate_next_dataset())
        data = np.asarray(data, dtype=np.float32)     # (B, N, D)
        graphs = np.asarray(graphs, dtype=np.float32) # (B, D, D)
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("label", data=graphs)
            f.attrs["has_labels"] = True
            f.attrs["num_nodes"] = num_vars
            f.attrs["num_samples"] = num_samples
            f.attrs["num_datasets"] = data.shape[0]
            f.attrs["function_generator"] = fg
        paths.append(path)
        if verbose:
            print(f"[gen] {os.path.basename(path)}: {data.shape[0]} datasets "
                  f"x {num_samples} samples x {num_vars} nodes  ({fg}), "
                  f"edges/graph={graphs.sum((1,2)).mean():.2f}")
    return paths


# ===========================================================================
# sanity eval: does the trained model recover the (known) edges?
# ===========================================================================
@torch.no_grad()
def sanity_eval(model, files, device="cpu", dtype=torch.float32, num_samples_forward=None):
    """AUC of predicted marginal edge prob vs true off-diagonal labels."""
    from . import bcnp_causal as B
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs_all, labels_all = [], []
    for path in files:
        data, label, _has, _names = B._read_hdf5(path)
        D = data.shape[-1]
        offdiag = ~np.eye(D, dtype=bool)
        for i in range(0, data.shape[0], 16):
            x = torch.tensor(np.nan_to_num(data[i:i + 16]), dtype=dtype, device=device)
            x = B._normalize_samples(x)
            p = model.forward(x, graph=None, is_training=False, mask=None).mean(0)  # (b,D,D)
            p = p.cpu().numpy()
            lab = label[i:i + 16]
            for b in range(p.shape[0]):
                probs_all.append(p[b][offdiag]); labels_all.append(lab[b][offdiag])
    probs_all = np.concatenate(probs_all); labels_all = np.concatenate(labels_all)
    if labels_all.max() == labels_all.min():
        return float("nan"), probs_all.mean()
    return float(roc_auc_score(labels_all, probs_all)), float(probs_all.mean())


# ===========================================================================
# main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description="Pretrain BCNP on synthetic labelled data (Option B, no TF)")
    p.add_argument("--num_vars", type=int, required=True,
                   help="node count; MUST match the D you later infer on (15 for PenSim 'sa')")
    p.add_argument("--csnp_root", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None,
                   help="where synthetic HDF5 go (default: datasets/data/synth_<D>var)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint path (default: pretrained/bcnp_<D>var.pt)")
    # generation
    p.add_argument("--generators", nargs="+", default=["gplvm", "neuralnet", "linear"],
                   choices=["gplvm", "neuralnet", "linear", "gplvm_neuralnet"])
    p.add_argument("--n_files", type=int, default=6)
    p.add_argument("--datasets_per_file", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--exp_edges_lower", type=float, default=0.0)
    p.add_argument("--exp_edges_upper", type=float, default=1.0)
    p.add_argument("--skip_gen", action="store_true", help="reuse existing HDF5 in data_dir")
    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers_encoder", type=int, default=4)
    p.add_argument("--num_layers_decoder", type=int, default=4)
    p.add_argument("--n_perm_samples", type=int, default=100)
    p.add_argument("--sinkhorn_iter", type=int, default=1000)
    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sanity", action="store_true", help="report edge-recovery AUC after training")
    p.add_argument("--n_test_files", type=int, default=4,
                   help="held-out synthetic files for a GENUINE generalization AUC "
                        "(0 = evaluate on training files instead)")
    p.add_argument("--eval_every", type=int, default=5,
                   help="epochs between held-out AUC checks DURING training (progress marker)")
    p.add_argument("--checkpoint_every", type=int, default=5,
                   help="epochs between checkpoint saves DURING training (crash-safety)")
    p.add_argument("--progress_log", type=str, default=None,
                   help="JSONL file to append progress markers to (default: <ckpt>.progress.jsonl)")
    p.add_argument("--no_resume", action="store_true",
                   help="ignore any existing <ckpt>.resume and train from scratch")
    args = p.parse_args()

    from . import bcnp_causal as B
    B._add_csnp_to_path(args.csnp_root)     # put BCNP repo on path (model import)

    data_dir = args.data_dir or f"datasets/data/synth_{args.num_vars}var"
    ckpt = args.ckpt or f"pretrained/bcnp_{args.num_vars}var.pt"

    # 1) synthetic data
    if args.skip_gen:
        files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        if not files:
            raise FileNotFoundError(f"--skip_gen but no HDF5 in {data_dir}")
        print(f"[gen] reusing {len(files)} existing files in {data_dir}")
    else:
        files = generate_synthetic(
            num_vars=args.num_vars, out_dir=data_dir, generators=tuple(args.generators),
            n_files=args.n_files, datasets_per_file=args.datasets_per_file,
            num_samples=args.num_samples, exp_edges_lower=args.exp_edges_lower,
            exp_edges_upper=args.exp_edges_upper, seed=args.seed,
        )

    # 2) held-out set (generated BEFORE training so it can be used as in-training
    #    eval_files -- progress markers, not just a post-hoc sanity check)
    test_files = None
    if args.n_test_files > 0:
        test_dir = os.path.join(data_dir, "heldout")
        print(f"[gen] generating {args.n_test_files} held-out files (unseen graphs)...")
        test_files = generate_synthetic(
            num_vars=args.num_vars, out_dir=test_dir, generators=tuple(args.generators),
            n_files=args.n_test_files, datasets_per_file=args.datasets_per_file,
            num_samples=args.num_samples, exp_edges_lower=args.exp_edges_lower,
            exp_edges_upper=args.exp_edges_upper, seed=args.seed + 100000,
        )

    # 3) model
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = B.build_model(
        num_nodes=args.num_vars, d_model=args.d_model, dim_feedforward=args.dim_feedforward,
        nhead=args.nhead, num_layers_encoder=args.num_layers_encoder,
        num_layers_decoder=args.num_layers_decoder, n_perm_samples=args.n_perm_samples,
        sinkhorn_iter=args.sinkhorn_iter, device=args.device,
    )
    print(f"[pretrain] num_nodes={args.num_vars} on {args.device}; "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # 4) train (reuse tested supervised loop) WITH progress markers
    progress_log = args.progress_log or (ckpt + ".progress.jsonl")
    print(f"[pretrain] progress markers -> {progress_log}  "
          f"(tail -f it to monitor a long/unattended run)")
    B.train_supervised(model, files, epochs=args.epochs, lr=args.lr,
                       batch_size=args.batch_size, device=args.device, save_path=ckpt,
                       eval_files=test_files, eval_every=args.eval_every,
                       checkpoint_every=args.checkpoint_every, progress_log=progress_log,
                       resume_path=ckpt + ".resume", resume=not args.no_resume)

    # 5) save meta (so infer uses matching architecture)
    meta = dict(num_nodes=args.num_vars, d_model=args.d_model,
                dim_feedforward=args.dim_feedforward, nhead=args.nhead,
                num_layers_encoder=args.num_layers_encoder,
                num_layers_decoder=args.num_layers_decoder,
                n_perm_samples=args.n_perm_samples, sinkhorn_iter=args.sinkhorn_iter,
                generators=args.generators, num_samples=args.num_samples)
    with open(ckpt + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pretrain] saved checkpoint -> {ckpt}\n[pretrain] saved meta -> {ckpt}.meta.json")

    # 6) final sanity (independent of in-training eval, same held-out files)
    if args.sanity:
        if test_files:
            auc, mean_p = sanity_eval(model, test_files, device=args.device)
            print(f"[sanity] FINAL held-out edge-recovery AUC = {auc:.3f}  "
                  f"(genuine generalization; 0.5 = chance). mean edge prob = {mean_p:.3f}")
        else:
            auc, mean_p = sanity_eval(model, files, device=args.device)
            print(f"[sanity] FINAL train-set edge-recovery AUC = {auc:.3f}  "
                  f"(fit, not generalization; pass --n_test_files>0 for held-out). "
                  f"mean edge prob = {mean_p:.3f}")

    print("\n[pretrain] to infer on PenSim, reuse the SAME architecture flags:")
    print(f"  poetry run python -m pensim_wm.bcnp_causal infer \\\n"
          f"    --data datasets/data/pensim_causal/pensim_phase0.hdf5 \\\n"
          f"    --checkpoint {ckpt} \\\n"
          f"    --d_model {args.d_model} --dim_feedforward {args.dim_feedforward} "
          f"--nhead {args.nhead} \\\n"
          f"    --num_layers_encoder {args.num_layers_encoder} "
          f"--num_layers_decoder {args.num_layers_decoder} \\\n"
          f"    --n_perm_samples {args.n_perm_samples} --sinkhorn_iter {args.sinkhorn_iter}")


if __name__ == "__main__":
    main()
