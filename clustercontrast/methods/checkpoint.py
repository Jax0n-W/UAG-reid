"""Training RNG state used when resuming CESA Stage 2 checkpoints."""

import random

import numpy as np
import torch


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1].copy()),
                  numpy_state[2], numpy_state[3], numpy_state[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    if state is None:
        return
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].numpy(),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
