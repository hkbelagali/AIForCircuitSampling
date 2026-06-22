"""Stage B CLI. Thin wrapper around aics.train_cell.

  python scripts/train.py --samples_npz <npz> --k_train 10000 --loss nll \\
      --out results/.../n12_k10000_h128_s0.json

  python scripts/train.py --samples_npz <npz> --k_train 2000 --loss z_pauli \\
      --curriculum weight_ascending --w_max 4 --out ...

Flag rules:
  --pt_regularizer  default ON for nll, FORBIDDEN for z_pauli
  --curriculum      z_pauli only
"""
import argparse

from aics import train_cell
from aics.io import save_result
from aics.runtime import print_hardware, assert_device_available, JsonLogger
from aics.training import (
    BATCH_SIZE, TOTAL_STEPS, MIN_EPOCHS, MAX_EPOCHS, LAMBDA_PT, SCHEDULES,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples_npz", required=True)
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
                    help="max weight for non-curriculum z_pauli (default n_qubits)")
    p.add_argument("--w_min", type=int, default=1)
    p.add_argument("--w_max", type=int, default=4)
    p.add_argument("--curriculum", choices=list(SCHEDULES) + ["none"], default="none")
    p.add_argument("--n_restarts_cold", type=int, default=4)
    p.add_argument("--n_restarts_warm", type=int, default=2)
    p.add_argument("--epochs_per_stage", type=int, default=400)
    p.add_argument("--device", default=None,
                    help="cpu | cuda | cuda:N (default: cpu unless --gpu)")
    p.add_argument("--gpu", action="store_true",
                    help="shortcut for --device cuda; hard error if no CUDA")
    p.add_argument("--resume", default=None, help="checkpoint to load before training")
    p.add_argument("--checkpoint", default=None, help="checkpoint to write at end")
    p.add_argument("--log_json", default=None, help="per-epoch JSON learning curve")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = assert_device_available(args.gpu, requested_device=args.device)
    print_hardware(device,
                    extra=f"loss={args.loss}  curriculum={args.curriculum}")

    logger = JsonLogger(args.log_json) if args.log_json else None
    result, _ = train_cell(
        args.samples_npz, k_train=args.k_train, model_seed=args.model_seed,
        hidden=args.hidden, n_layers=args.n_layers,
        loss=args.loss, pt_regularizer=args.pt_regularizer,
        pt_lambda=args.pt_lambda, lr=args.lr,
        total_steps=args.total_steps,
        min_epochs=args.min_epochs, max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        w_train=args.w_train, curriculum=args.curriculum,
        w_min=args.w_min, w_max=args.w_max,
        n_restarts_cold=args.n_restarts_cold,
        n_restarts_warm=args.n_restarts_warm,
        epochs_per_stage=args.epochs_per_stage,
        device=device, logger=logger,
        resume_from=args.resume, save_to=args.checkpoint, verbose=True,
    )
    if logger is not None:
        logger.close()

    save_result(args.out, result)
    print(f"[train] wrote {args.out}", flush=True)
    for k in ("loss", "final_nll", "final_loss", "held_nll", "xeb_gen", "xeb_norm"):
        if k in result:
            v = result[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}",
                  flush=True)


if __name__ == "__main__":
    main()
