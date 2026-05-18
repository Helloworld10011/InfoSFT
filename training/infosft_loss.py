import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def infosft_normalized_loss(outputs, labels, P, num_items_in_batch=None):
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = - logprobs.exp().detach() * torch.relu(1 - (logprobs.detach() - (1-logprobs.exp()).log().detach())/torch.log(torch.tensor(P/(1-P))).to(logprobs.device)) * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


def dft_loss(outputs, labels, num_items_in_batch=None):
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss
