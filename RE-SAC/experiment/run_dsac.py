"""
Wrapper for DSAC's dsac.py that injects gym_patch before import.
Run as: python run_dsac.py --config configs/dsac-normal-iqn-neutral/hopper.yaml --gpu 0 --seed 0
"""
import sys
import os

EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_ROOT)
import gym_patch  # noqa: F401

DSAC_ROOT = os.path.abspath(os.path.join(EXPERIMENT_ROOT, '../../../dsac'))
sys.path.insert(0, DSAC_ROOT)
os.chdir(DSAC_ROOT)

import rlkit.envs.vecenv as vecenv_module
_orig_VectorEnv_init = vecenv_module.VectorEnv.__init__

def _patched_VectorEnv_init(self, env_fns):
    _orig_VectorEnv_init(self, env_fns)
    if not hasattr(self.action_space, 'seed'):
        self.action_space.seed = lambda s=None: [s]

vecenv_module.VectorEnv.__init__ = _patched_VectorEnv_init

import runpy
runpy.run_path(os.path.join(DSAC_ROOT, 'dsac.py'), run_name='__main__')
