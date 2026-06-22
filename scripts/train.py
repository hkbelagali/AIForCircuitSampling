"""Stage B: train AutoregressiveRNN on a sample bundle, write result JSON.

  python scripts/train.py --samples_npz <npz> --k_train 10000 --loss nll \\
      --out results/.../n12_k10000_h128_s0.json

  python scripts/train.py --samples_npz <npz> --k_train 2000 --loss z_pauli \\
      --curriculum weight_ascending --w_max 4 --out ...

Flag rules:
  --pt_regularizer  default ON for nll, FORBIDDEN for z_pauli
  --curriculum      z_pauli only
"""
import argparse
import sys

import numpy as np
import torch

from aics.io import load_samples, save_result, bits_to_int
from aics.models import AutoregressiveRNN
from aics.runtime import (
    print_hardware, assert_device_available,
    save_checkpoint, load_checkpoint, JsonLogger,
)
from aics.training import (
    train_nll, train_z_pauli,
    BATCH_SIZE, TOTAL_STEPS, MIN_EPOCHS, MAX_EPOCHS, LAMBDA_PT, SCHEDULES,
)
from aics.training.curriculum import weight_ascending
from aics.eval import enumerate_z_supports, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples_npz", type=str, required=True)
    p.add_argument("--k_train", type=int, required=True)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--loss", choices=["nll", "z_pauli"], default="nll")
    p.add_argument("--pt_regularizer", action=argparse.BooleanOptionalAction,
                    default=None)
    p.add_argument("--pt_lambda", type=float, default=LAMBDA_PT)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--min_epochs", type=int, default=MIN_EPOCHS)
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--w_train", type=int, default=None,
                    help="max weight for non-curriculum z_pauli (default n)")
    p.add_argument("--w_min", type=int, default=1)
    p.add_argument("--w_max", type=int, default=4)
    p.add_argument("--curriculum", choices=list(SCHEDULES) + ["none"], default="none")
    p.add_argument("--n_restarts_cold", type=int, default=4)
    p.add_argument("--n_restarts_warm", type=int, default=2)
    p.add_argument("--epochs_per_stage", type=int, default=400)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--log_json", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.loss == "nll" and args.curriculum != "none":
        sys.exit("error: --curriculum is only valid with --loss z_pauli.")
    if args.loss == "z_pauli" and args.pt_regularizer is True:
        sys.exit("error: --pt_regularizer is not valid with --loss z_pauli.")
    if args.pt_regularizer is None:
        args.pt_regularizer = (args.loss == "nll")

    device = assert_device_available(args.gpu)
    print_hardware(device,
                    extra=f"loss={args.loss}  curriculum={args.curriculum}")

    data = load_samples(args.samples_npz)
    meta = data["meta"]
    n_qubits = meta["n"]   # meta key kept as "n" for back-compat with existing npz files
    D = 1 << n_qubits
    if args.k_train > len(data["train_bits"]):
        sys.exit(f"--k_train={args.k_train} exceeds k_max={len(data['train_bits'])}")
    train_bits = data["train_bits"][:args.k_train]
    held_bits = data.get("held_bits")
    held_pC = data.get("held_pC")
    uniform_pC = data.get("uniform_pC")

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)

    logger = JsonLogger(args.log_json) if args.log_json else None
    out = {
        "samples_npz": str(args.samples_npz),
        "n": n_qubits, "depth": meta["depth"], "circuit": meta.get("circuit"),
        "circuit_seed": meta["circuit_seed"], "sampler": meta.get("sampler"),
        "k_train": args.k_train, "model_seed": args.model_seed,
        "hidden": args.hidden, "n_layers": args.n_layers,
        "loss": args.loss, "pt_regularizer": args.pt_regularizer,
        "pt_lambda": args.pt_lambda if args.pt_regularizer else 0.0,
        "lr": args.lr, "curriculum": args.curriculum,
    }

    if args.loss == "nll":
        model = AutoregressiveRNN(n_bits=n_qubits, hidden=args.hidden,
                                    n_layers=args.n_layers).to(device)
        if args.resume:
            payload = load_checkpoint(args.resume, model, map_location=device)
            print(f"[train] resumed from {args.resume}  "
                  f"epoch={payload.get('epoch')}", flush=True)
        lam = args.pt_lambda if args.pt_regularizer else 0.0
        final_nll, n_epochs = train_nll(
            model, train_bits.astype(np.float32),
            total_steps=args.total_steps,
            min_epochs=args.min_epochs, max_epochs=args.max_epochs,
            batch_size=args.batch_size, lr=args.lr,
            lambda_pt=lam, n_states=D,
            device=device, verbose=False, logger=logger,
        )
        out["final_nll"] = final_nll
        out["n_epochs"] = n_epochs

    else:  # z_pauli
        samples_int = bits_to_int(train_bits)
        if args.curriculum == "weight_ascending":
            def _factory(seed):
                torch.manual_seed(seed)
                return AutoregressiveRNN(n_bits=n_qubits, hidden=args.hidden,
                                          n_layers=args.n_layers)
            stages = weight_ascending(
                _factory, samples_int, n_qubits,
                w_min=args.w_min, w_max=args.w_max,
                n_restarts_cold=args.n_restarts_cold,
                n_restarts_warm=args.n_restarts_warm,
                epochs_per_stage=args.epochs_per_stage,
                lr=args.lr, seed=args.model_seed,
                device=device, logger=logger, verbose=True,
            )
            last = stages[max(stages)]
            model = AutoregressiveRNN(n_bits=n_qubits, hidden=args.hidden,
                                        n_layers=args.n_layers).to(device)
            model.load_state_dict(last["model_state"])
            out["curriculum_stages"] = {
                k: {"w": v["w"], "best_loss": v["best_loss"],
                    "n_restarts": v["n_restarts"]}
                for k, v in stages.items()}
            out["final_loss"] = last["best_loss"]
        else:
            w_train = args.w_train if args.w_train is not None else n_qubits
            supports, weights = enumerate_z_supports(n_qubits, max_weight=w_train)
            model = AutoregressiveRNN(n_bits=n_qubits, hidden=args.hidden,
                                        n_layers=args.n_layers).to(device)
            if args.resume:
                load_checkpoint(args.resume, model, map_location=device)
            out["final_loss"] = train_z_pauli(
                model, samples_int, supports, weights, n_qubits,
                epochs=args.epochs_per_stage, lr=args.lr,
                device=device, verbose=True, logger=logger,
            )
            out["w_train"] = w_train

    model.eval()
    out.update(report(model, held_bits=held_bits, held_pC=held_pC,
                       uniform_pC=uniform_pC, n_qubits=n_qubits, device=device))

    if args.checkpoint:
        save_checkpoint(args.checkpoint, model)
        out["checkpoint"] = args.checkpoint
    if logger is not None:
        logger.close()

    save_result(args.out, out)
    print(f"[train] wrote {args.out}", flush=True)
    for k in ("loss", "final_nll", "final_loss", "held_nll", "xeb_gen", "xeb_norm"):
        if k in out:
            v = out[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}",
                  flush=True)


if __name__ == "__main__":
    main()
