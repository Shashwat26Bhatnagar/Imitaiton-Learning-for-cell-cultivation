import numpy as np, pandas as pd, sys
from pathlib import Path

# .../Dissertation/sindy-rl/sindy_rl/pensim/keig_sweep.py -> .../Dissertation
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'MCPILCO'))   # for gpr_lib

from gpr_lib.Utils import manifold_geometry as mg

CSV = ROOT / 'smpl' / 'smpl' / 'configdata' / 'pensimenv' / 'random_batch_0.csv'
assert CSV.exists(), f'not found: {CSV}'

ACT = ['Discharge rate', 'Sugar feed rate', 'Soil bean feed rate',
       'Aeration rate', 'Back pressure', 'Water injection/dilution']
ST  = ['pH', 'Temperature', 'Acid flow rate', 'Base flow rate', 'Cooling water',
       'Heating water', 'Vessel Weight', 'Dissolved oxygen concentration', 'Yield Per Step']

df = pd.read_csv(CSV)
s, a = df[ST].values, df[ACT].values
ds = s[1:] - s[:-1]
X  = np.concatenate([s[:-1], a[:-1]], 1)
MU, SD = X.mean(0), X.std(0) + 1e-8
Xs = (X - MU) / SD
V  = np.zeros((len(X), 15)); V[:, :9] = ds
Vs = V / SD

print(f"{'k_eig':>6} {'lam_min':>8} {'lam_max':>8} {'spread':>7} {'representable':>14}")
for k in (30, 50, 80, 120, 180):
    enc = mg.build_manifold_encoding_vector(       # ONE call, reused
        Xs, k_graph=15, k_eig=k, n_sub=200, normalized=False, verbose=False)
    Phi = enc["evecs"]
    y   = Vs[enc["sub_idx"]].reshape(-1, 1)
    c, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    cap = 1 - ((y - Phi @ c) ** 2).sum() / (y ** 2).sum()
    ev  = enc["evals"]
    print(f"{k:>6} {ev[0]:>8.3f} {ev[-1]:>8.3f} {ev[-1]/ev[0]:>6.1f}x {cap:>14.4f}")

print(f"\nceiling (rho^2) = 0.867")
