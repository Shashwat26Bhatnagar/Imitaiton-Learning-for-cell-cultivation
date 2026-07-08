#!/bin/bash
# BCNP synthetic pretraining at D=24 (num_vars=24 = "sas": [s_t(9), a_t(6), s_{t+1}(9)]).
#
# Submit:  sbatch pretrain_bcnp_d24.slurm
# Monitor: tail -f /work/m25oc/m25oc/s2892016/project/Dissertation/pretrained/bcnp_24var.pt.progress.jsonl
# Cancel:  scancel <JOBID>
#
# Compute nodes cannot see /home, and the Cray-Python shared lib
# (libpython3.10.so.1.0) must be on LD_LIBRARY_PATH for the venv python to load.
# We therefore: init the module system, load cray-python, and add its lib dir
# to LD_LIBRARY_PATH explicitly. No poetry, no ~/.bashrc, no /home paths.

#SBATCH --job-name=bcnp_d24
#SBATCH --account=m25oc-s2892016
#SBATCH --partition=gpu
#SBATCH --qos=gpu-exc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/work/m25oc/m25oc/s2892016/project/Dissertation/MCPILCO/pensim_wm/slurm/logs/bcnp_d24_%j.out
#SBATCH --error=/work/m25oc/m25oc/s2892016/project/Dissertation/MCPILCO/pensim_wm/slurm/logs/bcnp_d24_%j.err

set -euo pipefail

WORKDIR=/work/m25oc/m25oc/s2892016/project/Dissertation
cd "$WORKDIR"

# --- environment ------------------------------------------------------------
# initialize the module system in the non-interactive batch shell, THEN load
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cray-python
module load boost/1.81.0

# make libpython3.10.so.1.0 findable on the compute node (belt-and-suspenders)
export LD_LIBRARY_PATH="/opt/cray/pe/python/3.10.10/lib:${LD_LIBRARY_PATH:-}"

export CSNP_ROOT="$WORKDIR/BayesianDAG"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

PYTHON="$WORKDIR/.venv/bin/python"

mkdir -p "$WORKDIR/pretrained" "$WORKDIR/datasets/data"

# --- debug: prove the interpreter + its shared library are reachable --------
echo "[slurm] job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "[slurm] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[slurm] libpython check:"
ls -la /opt/cray/pe/python/3.10.10/lib/libpython3.10.so* 2>&1 | head -3
echo "[slurm] python: $PYTHON"
"$PYTHON" -c "import sys; print('[slurm] python OK', sys.version.split()[0])"
"$PYTHON" -c "import torch; print('[slurm] torch', torch.__version__)"

# --- run --------------------------------------------------------------------
# First real D=24 pass: neuralnet+linear (skip slow gplvm), modest size, frequent
# progress markers. Bump n_files/epochs and add gplvm once AUC is confirmed rising.
"$PYTHON" -m pensim_wm.pretrain_bcnp_synthetic \
    --num_vars 24 \
    --csnp_root "$CSNP_ROOT" \
    --generators neuralnet linear \
    --n_files 16 --datasets_per_file 128 --num_samples 300 \
    --d_model 128 --dim_feedforward 256 --nhead 8 \
    --num_layers_encoder 4 --num_layers_decoder 4 \
    --n_perm_samples 50 --sinkhorn_iter 300 \
    --epochs 20 --lr 1e-4 --batch_size 32 \
    --n_test_files 4 --eval_every 2 --checkpoint_every 2 \
    --data_dir "$WORKDIR/datasets/data/synth_24var" \
    --ckpt "$WORKDIR/pretrained/bcnp_24var.pt" \
    --progress_log "$WORKDIR/pretrained/bcnp_24var.pt.progress.jsonl" \
    --sanity

echo "[slurm] job ${SLURM_JOB_ID} finished at $(date)"
