"""Notebook-friendly Stage B helper. Mirrors scripts/train.py but returns
the result dict directly instead of writing JSON.

Ryan's notebook stays "pure" (uses internals directly even where duplicated);
this is for other notebooks that just want to drive the canonical pipeline.

Example:

    from aics import train_cell
    result, model = train_cell(
        "results/tn_samples/n12_d10_cs42_ss0_k100000.npz",
        k_train=10_000, hidden=128, loss="nll")
    print(result["xeb_norm"])
"""
import numpy as np
import torch

from .io import load_samples, bits_to_int
from .models import AutoregressiveRNN
from .runtime import load_checkpoint, save_checkpoint
from .training import train_nll, train_z_pauli, LAMBDA_PT, SCHEDULES
from .training.curriculum import weight_ascending
from .eval import enumerate_z_supports, report


def train_cell(samples_npz, *, k_train, hidden=128, n_layers=2,
                loss="nll", pt_regularizer=None, pt_lambda=LAMBDA_PT,
                lr=1e-3, total_steps=50_000, min_epochs=50, max_epochs=5_000,
                batch_size=512,
                # z_pauli-specific
                w_train=None, curriculum="none", w_min=1, w_max=4,
                n_restarts_cold=4, n_restarts_warm=2, epochs_per_stage=400,
                model_seed=0, device="cpu", logger=None,
                resume_from=None, save_to=None, verbose=False):
    """One Stage B cell. Returns (result_dict, model).

    Same flag rules as scripts/train.py:
      pt_regularizer  default ON for nll, REJECTED for z_pauli
      curriculum      z_pauli only; valid values from SCHEDULES

    resume_from   path to checkpoint to load before training (optional)
    save_to       path to write final checkpoint (optional)
    """
    if loss == "nll" and curriculum != "none":
        raise ValueError("curriculum is only valid with loss='z_pauli'")
    if loss == "z_pauli" and pt_regularizer is True:
        raise ValueError("pt_regularizer is not valid with loss='z_pauli'")
    if pt_regularizer is None:
        pt_regularizer = (loss == "nll")
    if curriculum != "none" and curriculum not in SCHEDULES:
        raise ValueError(f"unknown curriculum: {curriculum!r}")

    data = load_samples(samples_npz)
    meta = data["meta"]
    n_qubits = meta["n"]
    D = 1 << n_qubits
    if k_train > len(data["train_bits"]):
        raise ValueError(
            f"k_train={k_train} exceeds k_max={len(data['train_bits'])}")
    train_bits = data["train_bits"][:k_train]

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    result = {
        "samples_npz": str(samples_npz),
        "n": n_qubits, "depth": meta["depth"], "circuit": meta.get("circuit"),
        "circuit_seed": meta["circuit_seed"], "sampler": meta.get("sampler"),
        "k_train": k_train, "model_seed": model_seed,
        "hidden": hidden, "n_layers": n_layers,
        "loss": loss, "pt_regularizer": pt_regularizer,
        "pt_lambda": pt_lambda if pt_regularizer else 0.0,
        "lr": lr, "curriculum": curriculum,
    }

    if loss == "nll":
        model = AutoregressiveRNN(n_bits=n_qubits, hidden=hidden,
                                    n_layers=n_layers).to(device)
        if resume_from:
            load_checkpoint(resume_from, model, map_location=device)
        lam = pt_lambda if pt_regularizer else 0.0
        final_nll, n_epochs = train_nll(
            model, train_bits.astype(np.float32),
            total_steps=total_steps,
            min_epochs=min_epochs, max_epochs=max_epochs,
            batch_size=batch_size, lr=lr, lambda_pt=lam, n_states=D,
            device=device, verbose=verbose, logger=logger,
        )
        result["final_nll"] = final_nll
        result["n_epochs"] = n_epochs
    else:  # z_pauli
        samples_int = bits_to_int(train_bits)
        if curriculum == "weight_ascending":
            def _factory(seed):
                torch.manual_seed(seed)
                return AutoregressiveRNN(n_bits=n_qubits, hidden=hidden,
                                          n_layers=n_layers)
            stages = weight_ascending(
                _factory, samples_int, n_qubits,
                w_min=w_min, w_max=w_max,
                n_restarts_cold=n_restarts_cold,
                n_restarts_warm=n_restarts_warm,
                epochs_per_stage=epochs_per_stage,
                lr=lr, seed=model_seed,
                device=device, logger=logger, verbose=verbose,
            )
            last = stages[max(stages)]
            model = AutoregressiveRNN(n_bits=n_qubits, hidden=hidden,
                                        n_layers=n_layers).to(device)
            model.load_state_dict(last["model_state"])
            result["curriculum_stages"] = {
                k: {"w": v["w"], "best_loss": v["best_loss"],
                    "n_restarts": v["n_restarts"]}
                for k, v in stages.items()}
            result["final_loss"] = last["best_loss"]
        else:
            w_used = w_train if w_train is not None else n_qubits
            supports, weights = enumerate_z_supports(n_qubits, max_weight=w_used)
            model = AutoregressiveRNN(n_bits=n_qubits, hidden=hidden,
                                        n_layers=n_layers).to(device)
            if resume_from:
                load_checkpoint(resume_from, model, map_location=device)
            result["final_loss"] = train_z_pauli(
                model, samples_int, supports, weights, n_qubits,
                epochs=epochs_per_stage, lr=lr,
                device=device, verbose=verbose, logger=logger,
            )
            result["w_train"] = w_used

    model.eval()
    result.update(report(
        model,
        held_bits=data.get("held_bits"),
        held_pC=data.get("held_pC"),
        uniform_pC=data.get("uniform_pC"),
        n_qubits=n_qubits, device=device,
    ))

    if save_to:
        save_checkpoint(save_to, model)
        result["checkpoint"] = save_to
    return result, model
