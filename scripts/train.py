"""Stage B: load a sample bundle, train an AutoregressiveRNN, evaluate.

Two training losses:
  --loss nll       — NLL on bitstrings + optional PT regulariser (default ON,
                      strength via --pt_lambda).
  --loss z_pauli   — fit <Z_S>_θ to empirical <Z_S> on a chosen support set.
                      Tractable only for n ≲ 20. PT regulariser is NOT
                      available here (no theoretical motivation under Z-loss).

Curriculum:
  --curriculum weight_ascending  — Z-Pauli only. Warm-starts from w_min to w_max.
                                    Rejected with --loss nll.

Resume:
  --resume PATH    — load checkpoint and pick up. Useful for curriculum
                      stages that crashed mid-way.

Examples
--------

  # Ryan's NLL pipeline, n=12, hidden=128, k_train=10k:
  python scripts/train.py --samples_npz results/tn_samples/n12_d10_cs42_ss0_k100000.npz \\
    --k_train 10000 --hidden 128 --loss nll

  # Z-Pauli curriculum at n=8, w_min=1 → w_max=4:
  python scripts/train.py --samples_npz results/tn_samples/n8_d10_cs0_ss0_k100000.npz \\
    --k_train 2000 --loss z_pauli --curriculum weight_ascending --w_max 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from aics.io import load_samples, save_result, bits_to_int
from aics.models import AutoregressiveRNN
from aics.training import (
    train_nll_pt, train_z_pauli,
    BATCH_SIZE, TOTAL_STEPS, MIN_EPOCHS, MAX_EPOCHS, LAMBDA_PT,
    print_hardware, assert_device_available,
    save_checkpoint, load_checkpoint, JsonLogger,
    SCHEDULES,
)
from aics.training.curriculum import weight_ascending
from aics.eval import (
    linear_xeb_from_bits, normalized_xeb, held_nll, normalised_nll, nll_excess,
    enumerate_z_supports, empirical_z_expectations,
    model_z_expectations, per_weight_rms_err, per_weight_rms_true,
    unique_fraction, top1_fraction,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples_npz", type=str, required=True)
    p.add_argument("--k_train", type=int, required=True,
                    help="number of training samples (slice of cached train set)")
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--loss", choices=["nll", "z_pauli"], default="nll")
    # NLL-specific
    p.add_argument("--pt_regularizer", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="on/off PT reg; defaults: ON for nll, FORBIDDEN for z_pauli")
    p.add_argument("--pt_lambda", type=float, default=LAMBDA_PT)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--min_epochs", type=int, default=MIN_EPOCHS)
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    # Z-Pauli-specific
    p.add_argument("--w_train", type=int, default=None,
                    help="max weight for non-curriculum z_pauli (defaults to n)")
    p.add_argument("--w_min", type=int, default=1, help="curriculum start")
    p.add_argument("--w_max", type=int, default=4, help="curriculum end")
    p.add_argument("--curriculum", choices=list(SCHEDULES) + ["none"],
                    default="none")
    p.add_argument("--n_restarts_cold", type=int, default=4)
    p.add_argument("--n_restarts_warm", type=int, default=2)
    p.add_argument("--epochs_per_stage", type=int, default=400)
    # Shared
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--resume", type=str, default=None,
                    help="checkpoint path to load before training")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="checkpoint path to write at end (and periodically)")
    p.add_argument("--log_json", type=str, default=None,
                    help="path for per-epoch JSON (one line per epoch)")
    p.add_argument("--out", type=str, required=True,
                    help="path for the result JSON")
    args = p.parse_args()

    # ---- Validate cross-flag constraints
    if args.loss == "nll" and args.curriculum != "none":
        sys.exit("error: --curriculum is only valid with --loss z_pauli "
                  "(no NLL-style schedules implemented).")
    if args.loss == "z_pauli" and args.pt_regularizer is True:
        sys.exit("error: --pt_regularizer is not valid with --loss z_pauli "
                  "(no theoretical motivation under Z-observable loss).")

    if args.pt_regularizer is None:
        # Default: ON for nll, OFF (forbidden) for z_pauli
        args.pt_regularizer = (args.loss == "nll")

    device = assert_device_available(args.gpu)
    print_hardware(device,
                    extra=f"loss={args.loss}  curriculum={args.curriculum}")

    # ---- Load data
    data = load_samples(args.samples_npz)
    meta = data["meta"]
    n = meta["n"]
    D = 1 << n
    if args.k_train > len(data["train_bits"]):
        sys.exit(f"--k_train={args.k_train} exceeds cached "
                  f"k_max={len(data['train_bits'])}")
    train_bits = data["train_bits"][:args.k_train]
    held_bits = data.get("held_bits")
    held_pC = data.get("held_pC")
    uniform_pC = data.get("uniform_pC")

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)

    logger = JsonLogger(args.log_json) if args.log_json else None
    out = {
        "samples_npz": str(args.samples_npz),
        "n": n, "depth": meta["depth"], "circuit": meta.get("circuit"),
        "circuit_seed": meta["circuit_seed"],
        "sampler": meta.get("sampler"),
        "k_train": args.k_train, "model_seed": args.model_seed,
        "hidden": args.hidden, "n_layers": args.n_layers,
        "loss": args.loss, "pt_regularizer": args.pt_regularizer,
        "pt_lambda": args.pt_lambda if args.pt_regularizer else 0.0,
        "lr": args.lr, "curriculum": args.curriculum,
    }

    if args.loss == "nll":
        model = AutoregressiveRNN(n_bits=n, hidden=args.hidden,
                                    n_layers=args.n_layers).to(device)
        if args.resume:
            payload = load_checkpoint(args.resume, model, map_location=device)
            print(f"[train] resumed from {args.resume}  "
                  f"epoch={payload.get('epoch')}  "
                  f"best_loss={payload.get('best_loss')}", flush=True)
        lam = args.pt_lambda if args.pt_regularizer else 0.0
        final_nll, n_epochs = train_nll_pt(
            model, train_bits.astype(np.float32),
            total_steps=args.total_steps,
            min_epochs=args.min_epochs, max_epochs=args.max_epochs,
            batch_size=args.batch_size, lr=args.lr,
            lambda_pt=lam, n_states=D,
            device=device, verbose=False, logger=logger,
        )
        out["final_nll"] = final_nll
        out["n_epochs"] = n_epochs

    elif args.loss == "z_pauli":
        if args.curriculum == "weight_ascending":
            samples_int = bits_to_int(train_bits)

            def _factory(seed):
                torch.manual_seed(seed)
                return AutoregressiveRNN(n_bits=n, hidden=args.hidden,
                                          n_layers=args.n_layers)
            stages = weight_ascending(
                _factory, samples_int, n,
                w_min=args.w_min, w_max=args.w_max,
                n_restarts_cold=args.n_restarts_cold,
                n_restarts_warm=args.n_restarts_warm,
                epochs_per_stage=args.epochs_per_stage,
                lr=args.lr, seed=args.model_seed,
                device=device, logger=logger, verbose=True,
            )
            # Last stage's best state is what we evaluate
            last = stages[max(stages)]
            model = AutoregressiveRNN(n_bits=n, hidden=args.hidden,
                                        n_layers=args.n_layers).to(device)
            model.load_state_dict(last["model_state"])
            out["curriculum_stages"] = {
                k: {"w": v["w"], "best_loss": v["best_loss"],
                    "n_restarts": v["n_restarts"]}
                for k, v in stages.items()
            }
            out["final_loss"] = last["best_loss"]
        else:
            w_train = args.w_train if args.w_train is not None else n
            supports, weights = enumerate_z_supports(n, max_weight=w_train)
            samples_int = bits_to_int(train_bits)
            model = AutoregressiveRNN(n_bits=n, hidden=args.hidden,
                                        n_layers=args.n_layers).to(device)
            if args.resume:
                load_checkpoint(args.resume, model, map_location=device)
            final_loss = train_z_pauli(
                model, samples_int, supports, weights, n,
                epochs=args.epochs_per_stage, lr=args.lr,
                device=device, verbose=True, logger=logger,
            )
            out["final_loss"] = final_loss
            out["w_train"] = w_train

    # ---- Eval
    model.eval()
    if held_bits is not None:
        with torch.no_grad():
            held_t = torch.from_numpy(held_bits.astype(np.float32)).to(device)
            log_q_held = model.log_prob(held_t).cpu().numpy()
        out["held_nll"] = held_nll(log_q_held)
        out["normalised_nll"] = normalised_nll(out["held_nll"], n)
        if held_pC is not None:
            out["nll_excess"] = nll_excess(out["held_nll"], held_pC)
            q_held = np.exp(log_q_held)
            out["xeb_gen"] = float(D * q_held.mean() - 1)
            out["xeb_held_cache"] = float(D * held_pC.mean() - 1)
            if uniform_pC is not None:
                out["xeb_uniform_cache"] = float(D * uniform_pC.mean() - 1)
                out["xeb_norm"] = normalized_xeb(
                    out["xeb_gen"], out["xeb_uniform_cache"],
                    out["xeb_held_cache"])

    if args.checkpoint:
        save_checkpoint(args.checkpoint, model)
        out["checkpoint"] = args.checkpoint

    if logger is not None:
        logger.close()

    save_result(args.out, out)
    print(f"[train] wrote {args.out}", flush=True)
    for k in ("loss", "final_nll", "final_loss", "held_nll", "xeb_gen",
              "xeb_norm"):
        if k in out:
            v = out[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}",
                  flush=True)


if __name__ == "__main__":
    main()
