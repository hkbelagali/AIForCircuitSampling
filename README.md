# AI for Circuit Sampling (`aics`)

Learning RCS output distributions from bitstring samples. Walk-through: [`notebooks/rcs_ml_experiment.ipynb`](notebooks/rcs_ml_experiment.ipynb).

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
  training/       nll (+ optional PT reg, torch.compile + bf16 + fused Adam on CUDA), z_pauli (+ optional curriculum)
  eval/           xeb, nll metrics, z_observables, diversity, entropy, diagnostics, report
  io/             bit/qubit conventions, .npz samples, .json results, provenance
  runtime.py      device check, hardware banner, checkpoint I/O, JsonLogger
  cell.py         train_cell() — notebook-friendly Stage B helper

scripts/          sample.py, train.py, plot.py, plus experiment-specific scripts
slurm/            example_e2e_{gpu,cpu}.sb
notebooks/        rcs_ml_experiment.ipynb
tests/            aics pytest suite (27 tests)
plots/            committed figures
archive/          historical code: prototype-v1/, src/, m9/ (deprecated)
```

## Quickstart (CLI)

```bash
python scripts/sample.py --n 12
python scripts/train.py \
    --samples_npz results/tn_samples/n12_d10_cs42_ss0_k100000.npz \
    --k_train 10000 --loss nll \
    --out results/m_rcs_nll/n12_k10000_h128_s0.json
python scripts/plot.py
```

`--gpu` requires CUDA (hard error if unavailable; no silent CPU fallback). Every script prints a hardware banner.

Z-observable training with curriculum:
```bash
python scripts/train.py --samples_npz <npz> --k_train 2000 \
    --loss z_pauli --curriculum weight_ascending --w_min 1 --w_max 4 \
    --out results/m_rcs_z_pauli/n8_k2000.json
```
`--pt_regularizer` is rejected with `--loss z_pauli`; `--curriculum` is rejected with `--loss nll`.

## Quickstart (notebook)

```python
from aics import train_cell

result, model = train_cell(
    "results/tn_samples/n12_d10_cs42_ss0_k100000.npz",
    k_train=10_000, hidden=128, loss="nll",
)
print(result["xeb_norm"])
```

Same flag rules as the CLI. Returns `(result_dict, model)`.

## Samplers

| Sampler | Method | Bias | When |
|---|---|---|---|
| `exact_tn` *(default)* | `quimb.Circuit.sample`: sequential marginal-conditional, lightcone-trimmed, marginal caching | exact | always |
| `chaotic` | `quimb.Circuit.sample_chaotic`: uniform prior on non-marginal qubits | **biased** when marginal < n | v1 baseline only |
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

## m9 (Hubbard sample complexity, deprecated)

The `m9` package and its [notebook](archive/m9/notebooks/hubbard_sample_complexity.ipynb) now live under `archive/m9/`. Still installs alongside `aics` for back-compat (`pip install -e .` picks both up). Tests: `pytest archive/m9/tests/`.

## Branches

- `main` — current state.
- `v1-backup` — frozen pre-reorganisation reference (all old prototype scripts).
