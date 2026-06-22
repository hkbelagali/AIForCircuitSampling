"""Porter-Thomas regularizer.

Adds   λ · E_{z ~ batch}[ D · q(z) − log q(z) ]
to the NLL loss to keep the model distribution anti-concentrated like p_C
for random circuits. Equivalent (up to additive constants) to penalising
the second moment of q against the PT value 2/D.

Motivated for NLL training only — see `aics.training.z_pauli` which does
not (and should not) expose this. The exposed flag in scripts/train.py
is therefore loss-conditional.
"""
import torch


def pt_term(log_q_batch: torch.Tensor, n_states: int) -> torch.Tensor:
    """Compute the PT regulariser term:  E[D·q − log q].

    `log_q_batch` is a (B,) tensor of log model probabilities.
    `n_states` = 2^n; cast to float to avoid int overflow at n >= 63.

    Returns a 0-dim tensor.
    """
    q_scaled = torch.exp(log_q_batch) * float(n_states)
    return (q_scaled - log_q_batch).mean()
