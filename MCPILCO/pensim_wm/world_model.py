"""
GP world model for PenSim.

Two components, kept separate on purpose:

  * PenSimStateModel  -- one GP per observation channel (OBS_DIM=9), each
    predicting the *delta* o_{t+1} - o_t. This is exactly MC-PILCO's
    ``Model_learning_RBF`` behaviour, so we reuse it rather than reimplement.

  * PenSimRewardModel -- a single GP that maps (o_t, a_t) -> yield-per-step r_t
    as an *absolute* value (reward is NOT a state, so no delta / no integration).

Both operate in standardized space (see scalers.py) so unit-lengthscale RBF
priors are well conditioned despite PenSim's wildly different channel scales.

"""

import os
import pickle
import sys

import numpy as np
import torch

# --- make MC-PILCO importable when run from the repo root -------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import model_learning.Model_learning as ML          # noqa: E402
import gpr_lib.GP_prior.Stationary_GP as SGP         # noqa: E402
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood  # noqa: E402

from . import config                                 # noqa: E402
from .scalers import StandardScaler                  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rbf_init_dict(input_dim, dtype, device, sigma_n_init=0.05,
                   flg_train_mean=False, mean_init=None):
    d = dict(
        active_dims=np.arange(input_dim),
        lengthscales_init=np.ones(input_dim),
        flg_train_lengthscales=True,
        lambda_init=np.ones(1),
        flg_train_lambda=True,
        sigma_n_init=sigma_n_init * np.ones(1),
        sigma_n_num=None,
        flg_train_sigma_n=True,
        dtype=dtype,
        device=device,
    )
    if flg_train_mean:
        d["mean_init"] = np.zeros(1) if mean_init is None else np.asarray(mean_init)
        d["flg_train_mean"] = True
    return d


def _opt_dict(lr=0.01, n_epoch=800, n_print=200):
    return dict(
        f_optimizer=f"lambda p: torch.optim.Adam(p, lr={lr})",
        criterion=Likelihood.Marginal_log_likelihood,
        N_epoch=n_epoch,
        N_epoch_print=n_print,
    )


# ===========================================================================
# STATE MODEL (9 delta GPs, via MC-PILCO Model_learning_RBF)
# ===========================================================================
class PenSimStateModel:
    def __init__(self, dtype=torch.float64, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device
        self.obs_scaler = StandardScaler()
        self.act_scaler = StandardScaler()
        self.model = None            # Model_learning_RBF
        self._gp_input_dim = config.OBS_DIM + config.ACTION_DIM

    # -- fit ------------------------------------------------------------------
    def fit(self, buffer, stride=10, lr=0.01, n_epoch=800, n_print=200):
        # Fit scalers on all raw obs/actions in the buffer.
        Xo, Xa, _, _ = buffer.stacked_transitions(stride=1)
        self.obs_scaler.fit(Xo)
        self.act_scaler.fit(Xa)

        init = _rbf_init_dict(self._gp_input_dim, self.dtype, self.device)
        self.model = ML.Model_learning_RBF(
            num_gp=config.OBS_DIM,
            init_dict_list=[init] * config.OBS_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        # Add each episode separately (deltas must not cross episode boundaries).
        for obs, acts_padded in buffer.episode_arrays(stride=stride):
            s = self.obs_scaler.transform(obs)
            a = self.act_scaler.transform(acts_padded)
            self.model.add_data(new_state_samples=s, new_input_samples=a)

        opt = _opt_dict(lr=lr, n_epoch=n_epoch, n_print=n_print)
        self.model.reinforce_model(optimization_opt_list=[opt] * config.OBS_DIM)
        return self

    # -- predict --------------------------------------------------------------
    @torch.no_grad()
    def predict_next(self, obs_raw, act_raw, particle_pred=False):
        """(obs_raw, act_raw) -> (next_obs_raw, delta_std_raw)."""
        obs_raw = np.atleast_2d(obs_raw)
        act_raw = np.atleast_2d(act_raw)
        s = torch.tensor(self.obs_scaler.transform(obs_raw), dtype=self.dtype, device=self.device)
        a = torch.tensor(self.act_scaler.transform(act_raw), dtype=self.dtype, device=self.device)
        next_s, delta_mean, delta_var = self.model.get_next_state(
            current_state=s, current_input=a, particle_pred=particle_pred
        )
        next_obs_raw = self.obs_scaler.inverse_transform(next_s.cpu().numpy())
        delta_std_raw = self.obs_scaler.inverse_transform_delta(
            torch.sqrt(torch.clamp(delta_var, min=0)).cpu().numpy()
        )
        return next_obs_raw, delta_std_raw

    # -- persistence ----------------------------------------------------------
    def state_dict(self):
        m = self.model
        return {
            "obs_scaler": self.obs_scaler.state_dict(),
            "act_scaler": self.act_scaler.state_dict(),
            "gp_inputs": m.gp_inputs.cpu(),
            "gp_output_list": [g.cpu() for g in m.gp_output_list],
            "norm_list": m.norm_list,
            "num_samples": m.num_samples,
            "dim_state": m.dim_state,
            "dim_input": m.dim_input,
            "gp_state_dicts": [gp.state_dict() for gp in m.gp_list],
        }

    def load_state_dict(self, sd):
        self.obs_scaler = StandardScaler.from_state_dict(sd["obs_scaler"])
        self.act_scaler = StandardScaler.from_state_dict(sd["act_scaler"])
        init = _rbf_init_dict(self._gp_input_dim, self.dtype, self.device)
        self.model = ML.Model_learning_RBF(
            num_gp=config.OBS_DIM,
            init_dict_list=[init] * config.OBS_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        m = self.model
        m.gp_inputs = sd["gp_inputs"].to(self.device)
        m.gp_output_list = [g.to(self.device) for g in sd["gp_output_list"]]
        m.norm_list = sd["norm_list"]
        m.num_samples = sd["num_samples"]
        m.dim_state = sd["dim_state"]
        m.dim_input = sd["dim_input"]
        for gp, gsd in zip(m.gp_list, sd["gp_state_dicts"]):
            gp.load_state_dict(gsd)
        with torch.no_grad():
            for k in range(m.num_gp):
                m.pretrain_gp(k)
        return self


# ===========================================================================
# REWARD MODEL (single GP -> absolute yield-per-step)
# ===========================================================================
class PenSimRewardModel:
    def __init__(self, dtype=torch.float64, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device
        self.in_scaler = StandardScaler()   # over concat(obs, act)
        self.out_scaler = StandardScaler()  # over reward
        self.gp = None
        self._X_tr = None                   # standardized training inputs (kept for get_estimate)
        self._Y_tr = None                   # standardized training outputs
        self._gp_input_dim = config.OBS_DIM + config.ACTION_DIM

    def _raw_to_gp_in(self, obs_raw, act_raw):
        X = np.concatenate([np.atleast_2d(obs_raw), np.atleast_2d(act_raw)], axis=1)
        return self.in_scaler.transform(X)

    # -- fit ------------------------------------------------------------------
    def fit(self, buffer, stride=10, max_points=800, lr=0.02, n_epoch=800, n_print=200):
        Xo, Xa, _, Yr = buffer.stacked_transitions(stride=stride, max_points=max_points)
        X_raw = np.concatenate([Xo, Xa], axis=1)
        Yr = Yr.reshape(-1, 1)

        self.in_scaler.fit(X_raw)
        self.out_scaler.fit(Yr)
        Xs = self.in_scaler.transform(X_raw)
        Ys = self.out_scaler.transform(Yr)

        self._X_tr = torch.tensor(Xs, dtype=self.dtype, device=self.device)
        self._Y_tr = torch.tensor(Ys, dtype=self.dtype, device=self.device)

        # trainable-mean RBF: reward has a non-zero, input-dependent baseline.
        init = _rbf_init_dict(self._gp_input_dim, self.dtype, self.device,
                              sigma_n_init=0.1, flg_train_mean=True)
        self.gp = SGP.RBF(**init)

        ds = torch.utils.data.TensorDataset(self._X_tr, self._Y_tr)
        dl = torch.utils.data.DataLoader(ds, batch_size=self._X_tr.shape[0], shuffle=False)
        self.gp.fit_model(
            trainloader=dl,
            optimizer=torch.optim.Adam(self.gp.parameters(), lr=lr),
            criterion=Likelihood.Marginal_log_likelihood(),
            N_epoch=n_epoch,
            N_epoch_print=n_print,
        )
        return self

    # -- predict --------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs_raw, act_raw):
        """(obs_raw, act_raw) -> (reward_mean_raw, reward_std_raw)."""
        Xs = torch.tensor(self._raw_to_gp_in(obs_raw, act_raw), dtype=self.dtype, device=self.device)
        mean_s, var_s, *_ = self.gp.get_estimate(
            X=self._X_tr, Y=self._Y_tr, X_test=Xs, flg_return_K_X_inv=False
        )
        mean_s = mean_s.cpu().numpy().reshape(-1, 1)
        std_s = torch.sqrt(torch.clamp(var_s, min=0)).cpu().numpy().reshape(-1, 1)
        mean_raw = self.out_scaler.inverse_transform(mean_s).reshape(-1)
        std_raw = self.out_scaler.inverse_transform_delta(std_s).reshape(-1)
        return mean_raw, std_raw

    # -- persistence ----------------------------------------------------------
    def state_dict(self):
        return {
            "in_scaler": self.in_scaler.state_dict(),
            "out_scaler": self.out_scaler.state_dict(),
            "X_tr": self._X_tr.cpu(),
            "Y_tr": self._Y_tr.cpu(),
            "gp_state_dict": self.gp.state_dict(),
        }

    def load_state_dict(self, sd):
        self.in_scaler = StandardScaler.from_state_dict(sd["in_scaler"])
        self.out_scaler = StandardScaler.from_state_dict(sd["out_scaler"])
        self._X_tr = sd["X_tr"].to(self.device)
        self._Y_tr = sd["Y_tr"].to(self.device)
        init = _rbf_init_dict(self._gp_input_dim, self.dtype, self.device,
                              sigma_n_init=0.1, flg_train_mean=True)
        self.gp = SGP.RBF(**init)
        self.gp.load_state_dict(sd["gp_state_dict"])
        return self


# ===========================================================================
# COMBINED WORLD MODEL
# ===========================================================================
class PenSimWorldModel:
    def __init__(self, dtype=torch.float64, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device
        self.state_model = PenSimStateModel(dtype, device)
        self.reward_model = PenSimRewardModel(dtype, device)

    # -- fit both -------------------------------------------------------------
    def fit(self, buffer, state_stride=10, reward_stride=10, reward_max_points=800,
            state_kwargs=None, reward_kwargs=None):
        state_kwargs = state_kwargs or {}
        reward_kwargs = reward_kwargs or {}
        print("[world-model] training state model (9 delta GPs)...")
        self.state_model.fit(buffer, stride=state_stride, **state_kwargs)
        print("[world-model] training reward model (yield-per-step GP)...")
        self.reward_model.fit(buffer, stride=reward_stride, max_points=reward_max_points, **reward_kwargs)
        return self

    # -- one-step -------------------------------------------------------------
    def predict_next(self, obs_raw, act_raw, particle_pred=False):
        return self.state_model.predict_next(obs_raw, act_raw, particle_pred=particle_pred)

    def predict_reward(self, obs_raw, act_raw):
        return self.reward_model.predict(obs_raw, act_raw)

    # -- multi-step rollout ---------------------------------------------------
    @torch.no_grad()
    def rollout(self, obs0_raw, actions_raw, particle_pred=False):
        """
        Roll the model forward under a fixed action sequence.

        actions_raw : (H, ACTION_DIM)
        returns dict with obs (H+1, OBS_DIM), rewards (H,), cum_reward (float).
        """
        actions_raw = np.atleast_2d(actions_raw)
        H = actions_raw.shape[0]
        obs = np.atleast_2d(obs0_raw).astype(np.float64)
        obs_traj = [obs.reshape(-1)]
        rewards = []
        for t in range(H):
            a = actions_raw[t:t + 1]
            r, _ = self.predict_reward(obs, a)
            rewards.append(float(r[0]))
            obs, _ = self.predict_next(obs, a, particle_pred=particle_pred)
            obs_traj.append(obs.reshape(-1))
        rewards = np.asarray(rewards)
        return {
            "obs": np.stack(obs_traj, 0),
            "rewards": rewards,
            "cum_reward": float(rewards.sum()),
        }

    # -- persistence ----------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "state_model": self.state_model.state_dict(),
                    "reward_model": self.reward_model.state_dict(),
                    "dtype": str(self.dtype),
                },
                f,
            )
        print(f"[world-model] saved -> {path}")

    @classmethod
    def load(cls, path, device=torch.device("cpu")):
        with open(path, "rb") as f:
            d = pickle.load(f)
        dtype = torch.float64 if "float64" in d.get("dtype", "float64") else torch.float32
        wm = cls(dtype=dtype, device=device)
        wm.state_model.load_state_dict(d["state_model"])
        wm.reward_model.load_state_dict(d["reward_model"])
        print(f"[world-model] loaded <- {path}")
        return wm
