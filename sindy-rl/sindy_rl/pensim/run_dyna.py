import warnings
warnings.filterwarnings('ignore')
import os, logging, argparse
import numpy as np
import yaml
import ray
from ray import tune, air
from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos

from sindy_rl import _parent_dir
from sindy_rl.pbt_dyna import dyna_sindy
from sindy_rl.policy import RandomPolicy

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(
        _parent_dir, 'sindy_rl', 'config_templates', 'dyna_pensim.yml'))
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    LOCAL_DIR = os.path.join(_parent_dir, 'ray_results', cfg['exp_dir'])

    logging.basicConfig()
    logging.getLogger('dyna-sindy').setLevel(logging.INFO)

    n_control = cfg['drl']['config']['environment']['env_config']['act_dim']
    cfg['off_policy_pi'] = RandomPolicy(low=-np.ones(n_control),
                                        high=np.ones(n_control),
                                        zero_hold_n=10,   # 10h holds, not 1h jitter
                                        seed=0)

    ray.init(address=os.environ.get('ip_head', None), logging_level=logging.ERROR)

    drl_class, drl_default = rllib_algos.get(cfg['drl']['class'])()

    tune.Tuner(
        tune.with_resources(dyna_sindy, drl_class.default_resource_request(drl_default)),
        param_space=cfg,
        run_config=air.RunConfig(local_dir=LOCAL_DIR, **cfg['ray_config']['run_config']),
        tune_config=tune.TuneConfig(**cfg['ray_config']['tune_config']),
    ).fit()
