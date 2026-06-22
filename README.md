# AI for Circuit Sampling (`aics`)

Learning RCS output distributions from bitstring samples. Walk-through: [`notebooks/rcs_ml_experiment.ipynb`](notebooks/rcs_ml_experiment.ipynb).

Companion: [`m9`](m9/) — Hubbard ground-state sample complexity ([`notebooks/hubbard_sample_complexity.ipynb`](notebooks/hubbard_sample_complexity.ipynb)). Both ship from this repo.

## Setup

```
pip install -e .
pytest tests/
```

## Layout

```
src/aics/
  circuits/       boixo_v2 (default, Ryan's), sycamore, exact (cirq reference)
  sampling/       exact_tn (default, unbiased), chaotic (biased baseline), amplitudes
  models/         AutoregressiveRNN (LSTM, hidden=128)
  training/       nll (+ optional PT reg), z_pauli (+ optional curriculum)
  eval/           xeb, nll metrics, z_observables, diversity, entropy
  io/             bit/qubit conventions, .npz samples, .json results, provenance

scripts/          sample.py, train.py, plot.py
slurm/            example_e2e_{gpu,cpu}.sb
notebooks/        rcs_ml_experiment.ipynb, hubbard_sample_complexity.ipynb
tests/            pytest suite
plots/            committed figures
archive/          historical code (prototype-v1/, src/)
m9/               Hubbard package
```

## Quickstart

```bash
python scripts/sample.py --n 12
python scripts/train.py \
    --samples_npz results/tn_samples/n12_d10_cs42_ss0_k100000.npz \
    --k_train 10000 --loss nll \
    --out results/m_rcs_nll/n12_k10000_h128_s0.json
python scripts/plot.py
```

`--gpu` requires CUDA (hard error if unavailable; no silent CPU fallback). All scripts print a hardware banner.

Z-observable training with curriculum:
```bash
python scripts/train.py --samples_npz <npz> --k_train 2000 \
    --loss z_pauli --curriculum weight_ascending --w_min 1 --w_max 4 \
    --out results/m_rcs_z_pauli/n8_k2000.json
```
`--pt_regularizer` is rejected with `--loss z_pauli`; `--curriculum` is rejected with `--loss nll`.

## Samplers

| Sampler | Method | Bias | When |
|---|---|---|---|
| `exact_tn` *(default)* | `quimb.Circuit.sample`: sequential marginal-conditional, lightcone-trimmed, marginal caching | exact | always |
| `chaotic` | `quimb.Circuit.sample_chaotic`: uniform prior on non-marginal qubits | **biased** when marginal < n | v1 comparison only |
| `sample_from_circuit` *(in `aics.circuits.exact`)* | cirq full statevector + multinomial | exact | n ≤ 26, used by tests |

At n=24 depth 10, `chaotic(marginal=20)` underestimates XEB by ~0.04 vs `exact_tn`.

## Reproducibility

Every `.npz` and `.json` artifact carries a `provenance` block: `{git_commit, timestamp_utc, hostname, pid, config}`. Trace any plot back to the exact code + flags that produced it.

## SLURM

```
sbatch slurm/example_e2e_gpu.sb
N=24 K=10000 H=256 sbatch slurm/example_e2e_gpu.sb
```

Templates only — copy and edit. Job-specific `.sb` files are not committed.

## Branches

- `main` — current state.
- `v1-backup` — frozen pre-reorganisation reference (all old prototype scripts).
