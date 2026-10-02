"""Shared strict checkpoint loading for final AGW evaluation."""


def load_agw_checkpoint_strict(model, checkpoint):
    state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError('Checkpoint architecture does not match AGW evaluation model: missing state_dict.')
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'Checkpoint architecture does not match AGW evaluation model.') from error
    return model
