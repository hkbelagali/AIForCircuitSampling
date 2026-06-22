"""Porter-Thomas regularizer added to NLL.

  λ · E[ D · q(z) − log q(z) ]

Penalises the second moment of q against the PT value 2/D. NLL-only —
not exposed under Z-observable loss.
"""
import torch


def pt_term(log_q_batch, n_states):
    """log_q_batch: (B,) tensor. n_states cast to float to dodge int overflow at n>=63."""
    q_scaled = torch.exp(log_q_batch) * float(n_states)
    return (q_scaled - log_q_batch).mean()
