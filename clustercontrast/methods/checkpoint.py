"""Fixed-final checkpoint policy and resumable training state helpers."""

import os
import random

import numpy as np
import torch


LATEST_CHECKPOINT = 'checkpoint.pth.tar'
FINAL_CHECKPOINT = 'model_final.pth.tar'
BEST_CHECKPOINT = 'model_best.pth.tar'
BEST_SELECTION_METRIC = 'g2a_rank1'


def final_checkpoint_path(directory):
    return os.path.join(directory, FINAL_CHECKPOINT)


def best_checkpoint_path(directory):
    return os.path.join(directory, BEST_CHECKPOINT)


def resolve_stage1_initialization(directory, source, stage2_resume=False):
    """Resolve the explicitly selected Stage1 source for a fresh Stage2."""
    if stage2_resume:
        return None
    if source == 'final':
        path = final_checkpoint_path(directory)
    elif source == 'best':
        path = best_checkpoint_path(directory)
    else:
        raise ValueError('Stage1 initialization must be final or best')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'Requested Stage1 {} checkpoint does not exist: {}. '
            'Use --eval-during-train=True to produce a best checkpoint.'
            .format(source, path))
    return path


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


def select_best_rank1(current_rank1, best_rank1, best_epoch, epoch):
    """Select the earliest epoch with the highest G-to-A Rank-1."""
    current_rank1 = float(current_rank1)
    if current_rank1 > float(best_rank1):
        return current_rank1, int(epoch), True
    return float(best_rank1), best_epoch, False


def select_agreid_best(eval_a2g, eval_g2a, best_rank1, best_epoch, epoch):
    """Select by G-to-A Rank-1; A-to-G is diagnostic only."""
    del eval_a2g
    return select_best_rank1(
        eval_g2a['rank1'], best_rank1, best_epoch, epoch)


def restore_best_state(checkpoint):
    """Load best metadata while accepting checkpoints from older releases."""
    return (float(checkpoint.get('best_R1', float('-inf'))),
            checkpoint.get('best_epoch'))


def save_best_checkpoint(state, directory):
    os.makedirs(directory, exist_ok=True)
    path = best_checkpoint_path(directory)
    torch.save(state, path)
    return path


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
