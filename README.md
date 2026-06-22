# AI for Circuit Sampling (`aics`)

Classical learnability of Random Circuit Sampling (RCS) output distributions, via an autoregressive RNN trained on bitstring samples. Walk-through in [`notebooks/rcs_ml_experiment.ipynb`](notebooks/rcs_ml_experiment.ipynb).

Companion project: [`m9`](m9/) — Hubbard ground-state sample complexity ([`notebooks/hubbard_sample_complexity.ipynb`](notebooks/hubbard_sample_complexity.ipynb)). Both packages ship from this repo.

## Setup

```
pip install -e .
```

This installs both `aics` and `m9`. Run `pytest tests/` to verify.

## What's in here

```
src/aics/
├── circuits/        # boixo_v2 (default, Ryan's circuit), sycamore, exact (cirq reference)
├── sampling/        # exact_tn (default unbiased, quimb sequential marginal),
│                    # chaotic (BIASED baseline, kept w/ loud warning), amplitudes
├── models/          # AutoregressiveRNN (Ryan's LSTM, hidden=128 by default)
├── training/        # nll (with PT regularizer flag), z_pauli (with curriculum), Trainer
├── eval/            # XEB, held-out NLL, Z-observables, diversity, entropy
└── io/              # bit/qubit conventions, sample-bundle npz I/O, JSON result I/O,
                     # automatic provenance (git commit + timestamp) on every artifact

scripts/             # CLI entry points: sample.py, train.py, plot.py
slurm/               # example end-to-end .sb templates (gpu + cpu variants)
notebooks/           # Ryan's RCS notebook + Hubbard sample-complexity notebook
tests/               # pytest suite — 26 tests covering circuits, model, training,
                     # eval, sampling parity (sample_exact_tn ↔ sample_exact_cirq)
plots/               # publication-ready figures (commit explicitly)
archive/             # historical code preserved verbatim
  prototype-v1/      # the v1 prototype/ tree before the src/aics/ reorganisation
  src/               # the original archive/src/aics scaffold (subsumed by src/aics/)
m9/                  # Hubbard sample-complexity (untouched, lives alongside aics)
```

## Quickstart — end-to-end at n=12, NLL training

```bash
# Stage A: build circuit, draw 100k samples, compute p_C(z), save npz
python scripts/sample.py --n 12

# Stage B: train RNN at k_train=10000, hidden=128, NLL + PT regulariser
python scripts/train.py \
    --samples_npz results/tn_samples/n12_d10_cs42_ss0_k100000.npz \
    --k_train 10000 \
    --loss nll \
    --out results/m_rcs_nll_aics/n12_k10000_h128_s0.json

# Plot
python scripts/plot.py
```

Add `--gpu` to either script to require CUDA (hard error if unavailable — no silent CPU fallback). Every script prints a hardware banner at start.

### Z-observable training with a weight-ascending curriculum

```bash
python scripts/train.py \
    --samples_npz results/tn_samples/n8_d10_cs42_ss0_k100000.npz \
    --k_train 2000 \
    --loss z_pauli \
    --curriculum weight_ascending --w_min 1 --w_max 4 \
    --out results/m_rcs_z_pauli/n8_k2000.json
```

`--pt_regularizer` is forbidden with `--loss z_pauli` (no theoretical motivation under Z-observable loss); `--curriculum` is forbidden with `--loss nll`. Both error with a clear message.

## Samplers

| Sampler | Cost | Bias | When to use |
|---|---|---|---|
| `exact_tn` *(default)* | sequential marginal-conditional, quimb lightcone-trimmed contraction with marginal caching | none (exact, modulo dtype precision) | always, unless you specifically want the v1 baseline for comparison |
| `chaotic` | one TN contraction + uniform random for non-marginal qubits | **biased** on non-PT distributions | v1 historical comparisons only — see the loud docstring warning |
| `sample_exact_cirq` *(in `aics.circuits.exact`)* | full cirq statevector + multinomial | none | n ≤ ~26 only; used by `tests/test_sampling_parity.py` to cross-check `exact_tn` |
| `tebd` *(planned)* | explicit MPS with truncation | bounded by truncation cutoff | future, when we want to study cutoff-vs-accuracy |

Bias quantification at n=24, depth 10: `chaotic(marginal=20)` underestimates XEB by ~0.04 vs `exact_tn` (Job 10693457 result). Bias grows with n; default to `exact_tn`.

## Reproducibility

Every Stage A `.npz` and every Stage B `.json` carries a `provenance` block embedded by `aics.io`:

```json
"provenance": {
  "git_commit": "9e92dec-dirty",
  "timestamp_utc": "2026-06-22T05:55:12Z",
  "hostname": "skl-027",
  "pid": 12345,
  "config": {...}
}
```

So any plot can be traced back to (a) the exact code that produced it and (b) the dataset bundle that fed it.

## SLURM

```
sbatch slurm/example_e2e_gpu.sb              # default n=12 on data-machine
N=24 K=10000 H=256 sbatch slurm/example_e2e_gpu.sb
```

The `.sb` files are intentional templates — copy and edit. No proliferation of one-off `submit_*.sb` in the repo root (the v1 ones were not committed).

## Tests

```
pytest tests/
```

The new RCS hot-loop suite is 26 tests (circuits, model, NLL + Z-Pauli training, eval metrics, exact_tn ↔ exact_cirq parity at small n). The pre-existing Hubbard suite (`test_components.py`, `test_smoke.py`) is unchanged.

## History

- `git checkout v1-backup` — frozen state immediately before the `src/aics/` reorganisation. Imports cleanly; everything points at `archive/prototype-v1/`.
- `git checkout main` — unchanged from `origin/main`; the reorg lives entirely on `cleanup/repo-restructure` until merge.
