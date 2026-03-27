"""
Wrapper for DSAC's sac.py that injects gym_patch before import.
Run as: python run_sac.py --config configs/sac-normal/hopper.yaml --gpu 0 --seed 0
"""
import sys
import os

# Must be first: patch gym.spaces.Box.seed() for old gym versions
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_ROOT)
import gym_patch  # noqa: F401

# Add DSAC root
DSAC_ROOT = os.path.abspath(os.path.join(EXPERIMENT_ROOT, '../../../dsac'))
sys.path.insert(0, DSAC_ROOT)
os.chdir(DSAC_ROOT)  # sac.py opens config paths relative to DSAC_ROOT

# Patch action_space.seed call (guard it)
import rlkit.envs.vecenv as vecenv_module
_orig_VectorEnv_init = vecenv_module.VectorEnv.__init__

def _patched_VectorEnv_init(self, env_fns):
    _orig_VectorEnv_init(self, env_fns)
    # Ensure action_space has .seed if missing
    if not hasattr(self.action_space, 'seed'):
        self.action_space.seed = lambda s=None: [s]

vecenv_module.VectorEnv.__init__ = _patched_VectorEnv_init

# Now run sac.py as __main__
import runpy
runpy.run_path(os.path.join(DSAC_ROOT, 'sac.py'), run_name='__main__')
