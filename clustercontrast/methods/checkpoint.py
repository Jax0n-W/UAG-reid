"""Fixed-final checkpoint policy and resumable training state helpers."""

import os
import random

import numpy as np
import torch


LATEST_CHECKPOINT = 'checkpoint.pth.tar'
FINAL_CHECKPOINT = 'model_final.pth.tar'


def final_checkpoint_path(directory):
    return os.path.join(directory, FINAL_CHECKPOINT)


def save_fixed_epoch_checkpoint(state, directory, is_final_epoch):
    """Always save latest state and also save the predetermined final epoch."""
    os.makedirs(directory, exist_ok=True)
    latest_path = os.path.join(directory, LATEST_CHECKPOINT)
    torch.save(state, latest_path)
    final_path = None
    if is_final_epoch:
        final_path = final_checkpoint_path(directory)
        torch.save(state, final_path)
    return latest_path, final_path


def should_evaluate_during_train(args, epoch):
    return (bool(args.eval_during_train)
            and (epoch + 1) % max(1, int(args.eval_step)) == 0)


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
