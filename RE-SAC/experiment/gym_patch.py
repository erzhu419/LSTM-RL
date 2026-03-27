"""
Compatibility patch for old gym versions missing Box.seed().

Import this module FIRST (before any rlkit imports) to monkey-patch
gym.spaces.Box with a no-op .seed() method.

Required for: gym 0.10.x - 0.20.x compatibility with DSAC codebase.
"""
import gym.spaces


def _box_seed(self, seed=None):
    """No-op seed method for gym.spaces.Box compatibility."""
    return [seed]


# Inject if missing
if not hasattr(gym.spaces.Box, 'seed'):
    gym.spaces.Box.seed = _box_seed

# Also patch Discrete and other spaces just in case
for space_class in [gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary]:
    if not hasattr(space_class, 'seed'):
        space_class.seed = _box_seed
