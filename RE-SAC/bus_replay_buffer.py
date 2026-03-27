import random
from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, last_episode_step: int = 5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step  # Estimated steps per episode for pruning
        self.buffer: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = {}
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position += 1

        if len(self.buffer) > self.capacity:
            keys_to_remove = list(self.buffer.keys())[:self.last_episode_step]
            for key in keys_to_remove:
                del self.buffer[key]

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer.values()), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
