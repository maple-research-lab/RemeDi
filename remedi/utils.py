import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except ImportError:
    DeepSpeedEngine = None  # Fallback in case DeepSpeed is not installed


def get_config(model: nn.Module):
    """
    Retrieve the `config` attribute from a model, whether it's a plain nn.Module,
    a DistributedDataParallel (DDP) model, or a DeepSpeed-wrapped model.
    """
    # Plain nn.Module
    if isinstance(model, nn.Module) and not isinstance(model, (DDP, DeepSpeedEngine if DeepSpeedEngine else tuple())):
        if hasattr(model, "config"):
            return model.config

    # DDP-wrapped model
    if isinstance(model, DDP):
        if hasattr(model.module, "config"):
            return model.module.config

    # DeepSpeed-wrapped model
    if DeepSpeedEngine and isinstance(model, DeepSpeedEngine):
        # DeepSpeed wraps the original model in `module`, similar to DDP
        if hasattr(model.module, "config"):
            return model.module.config

    raise AttributeError("The model (including its wrappers) does not have a `config` attribute.")


def reverse_cumsum(x, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim=dim), [dim])


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        input_shape = categorical_probs.shape
        N = input_shape[-1]
        flat_probs = categorical_probs.reshape(-1, N)
        samples = torch.multinomial(flat_probs, num_samples=1).squeeze(-1) 
        return samples.reshape(input_shape[:-1])
    elif method == 'max':
        return categorical_probs.argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


def get_pt_logprobs(remask_prob, sample_order):
    batch_size = remask_prob.shape[0]
    bs_idx = torch.arange(batch_size, dtype=sample_order.dtype).unsqueeze(1)
    sample_flag = torch.zeros_like(remask_prob, dtype=torch.bool)
    sample_flag[bs_idx, sample_order] = True

    # F-set sum
    sum_neg = (remask_prob * (~sample_flag)).sum(-1, keepdim=True) # bs, 1
    # (bs, gen_length // steps)
    select_remask = remask_prob[bs_idx, sample_order]
    # (bs, 1)  + (bs, gen_length // steps)
    fen_mu_log = (sum_neg + reverse_cumsum(select_remask, dim=-1)).log()
    pt_logprobs = select_remask.log() - fen_mu_log
    return pt_logprobs


def interpolate_p(prob, x0, mask_id=126336, alpha=0):
    """
    Interpolate between the probability distribution `prob` and a one-hot distribution
    based on whether each token in `x0` is equal to `mask_id`.

    Args:
        prob (FloatTensor): shape (bs, len, vocab) — predicted probability distributions.
        x0 (LongTensor): shape (bs, len) — token ids.
        mask_id (int): ID to compare in x0 to determine interpolation behavior.
        alpha (float): interpolation weight (0 = only one-hot, 1 = only prob).

    Returns:
        output_prob (FloatTensor): interpolated output distribution, shape (bs, len, vocab).
    """
    bs, length, vocab = prob.shape

    # Create one-hot from x0
    one_hot = F.one_hot(x0, num_classes=vocab).float()

    # Condition for where to apply interpolation
    mask = (x0 != mask_id).unsqueeze(-1)  # shape (bs, len, 1)

    # Interpolation
    output_prob = torch.where(
        mask,  # where x0 != mask_id
        alpha * prob + (1 - alpha) * one_hot,
        prob  # where x0 == mask_id
    )

    return output_prob
