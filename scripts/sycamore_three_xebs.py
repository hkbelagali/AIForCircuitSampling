"""Given a trained model checkpoint, report all three XEBs:

  device XEB           = D * E_{z~experimental}[p_ideal(z)] - 1   (fixed baseline)
  device XEB trained   = D * E_{z~experimental}[q_model(z)] - 1   (model on device samples)
  clean XEB            = D * E_{z~ideal}[q_model(z)] - 1          (model on ideal samples)

The device baseline lives in device_held.npz meta. Both evals use the same
model checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aics.io import load_samples
from aics.models import AutoregressiveRNN
from aics.runtime import assert_device_available


def load_model(checkpoint, hidden, n_qubits, device):
    payload = torch.load(checkpoint, map_location=device)
    m = AutoregressiveRNN(n_bits=n_qubits, hidden=hidden, n_layers=2).to(device)
    m.load_state_dict(payload["model_state"])
    m.eval()
    return m, payload.get("epoch", "?")


@torch.no_grad()
def xeb_on_held(model, samples_npz, device):
    """Return unnormalized xeb = D * <q_model(z_held)> - 1."""
    data = load_samples(samples_npz)
    n = data["meta"]["n"]
    D = 1 << n
    held = torch.from_numpy(np.asarray(data["held_bits"], dtype=np.float32)).to(device)
    log_q = model.log_prob(held).cpu().numpy()
    q = np.exp(log_q)
    return float(D * q.mean() - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clean_npz", required=True,
                    help="npz whose held_bits are ideal-TN samples (e.g. exp_pool or tn_pool)")
    p.add_argument("--device_npz", required=True,
                    help="npz whose held_bits are experimental measurements with p_ideal")
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    device = assert_device_available(args.gpu, requested_device=args.device)

    clean_data = load_samples(args.clean_npz)
    device_data = load_samples(args.device_npz)
    n = clean_data["meta"]["n"]
    device_baseline = float(device_data["meta"].get("device_xeb_baseline",
                                                      float("nan")))

    model, epoch = load_model(args.checkpoint, args.hidden, n, device)

    clean_xeb = xeb_on_held(model, args.clean_npz, device)
    device_xeb_trained = xeb_on_held(model, args.device_npz, device)

    result = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "n_qubits": n,
        "device_xeb": device_baseline,
        "device_xeb_trained": device_xeb_trained,
        "clean_xeb": clean_xeb,
        "denoising_ratio": (device_xeb_trained - device_baseline) if np.isfinite(device_baseline) else None,
    }
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
