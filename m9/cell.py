"""Run one (L, k, seed) cell and return the steps-to-threshold record."""

import numpy as np
import torch

from .hubbard import Hubbard
from .model import ARRNN
from .training import model_energy_exact, train_mle, vmc_step


def run_cell(L, k, seed, *,
             U=4.0, threshold=0.01, max_steps=5000, eval_every=10,
             pretrain_epochs=100, vmc_lr=3e-3, vmc_batch=256, d_hidden=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = Hubbard(L, U=U)
    model = ARRNN(L, L // 2, L // 2, d_hidden=d_hidden)

    if k > 0:
        train_mle(model, ctx.sample(k, rng), epochs=pretrain_epochs)

    opt = torch.optim.Adam(model.parameters(), lr=vmc_lr)
    steps = [0]
    energies = [model_energy_exact(model, ctx)]
    rel0 = (energies[0] - ctx.E_0) / abs(ctx.E_0)
    steps_to_threshold = 0 if abs(rel0) <= threshold else None

    for step in range(1, max_steps + 1):
        vmc_step(model, ctx, opt, batch=vmc_batch)
        if step % eval_every == 0 or step == max_steps:
            E = model_energy_exact(model, ctx)
            steps.append(step); energies.append(E)
            rel = (E - ctx.E_0) / abs(ctx.E_0)
            if steps_to_threshold is None and abs(rel) <= threshold:
                steps_to_threshold = step
                break

    return {
        "L": L, "k": k, "seed": seed, "U": U,
        "E_0": ctx.E_0, "steps": steps, "energies": energies,
        "steps_to_threshold": steps_to_threshold,
        "max_steps": max_steps,
    }
