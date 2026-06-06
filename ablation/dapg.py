from __future__ import annotations

from typing import Any

from offline2online.offline_data import ExpertReplayBuffer, compute_bc_loss


def compute_dapg_demo_loss(agent: Any, expert_buffer: ExpertReplayBuffer, batch_size: int, device):
    """Return the DAPG demonstration log-likelihood loss.

    This is intentionally a thin baseline hook. The trainer decides how to
    weight, decay, and combine this demonstration gradient with the on-policy
    PPO gradient, matching the DAPG-style comparison protocol.
    """
    return compute_bc_loss(agent, expert_buffer, batch_size=batch_size, device=device)
