"""Component tests: Hubbard ED, local energy consistency, sampling, signs."""

import numpy as np
import pytest
import torch

from m9 import Hubbard, state_int_to_bits, local_energy, ARRNN


@pytest.mark.parametrize("L", [4, 6])
def test_hubbard_ed_eigenvalue(L):
    ctx = Hubbard(L, U=4.0)
    residual = ctx.H @ ctx.psi_0 - ctx.E_0 * ctx.psi_0
    assert np.linalg.norm(residual) < 1e-8
    assert np.allclose(ctx.psi_0 @ ctx.psi_0, 1.0)


@pytest.mark.parametrize("L", [4, 6])
def test_local_energy_at_GS(L):
    """Substitute exact log|psi_0| for the model; E_loc(x) must equal E_0 everywhere."""
    ctx = Hubbard(L, U=4.0)
    abs_psi = np.abs(ctx.psi_0)

    class OracleModel:
        def log_psi(self, bits_t):
            ints = bits_t.cpu().numpy() @ (1 << np.arange(2 * L, dtype=np.int64))
            log_abs = np.log(np.maximum(abs_psi[np.array([ctx.idx[int(i)] for i in ints])],
                                        1e-300))
            return torch.from_numpy(log_abs)

    # Take all sector states; E_loc(x) should equal E_0 for each (since psi=GS).
    bits = state_int_to_bits(ctx.states, L)
    E_loc = local_energy(OracleModel(), bits, ctx).numpy()
    # Only states with |psi| > 0 have well-defined E_loc; the GS has full support
    # in our convention so this checks every state.
    nonzero = abs_psi > 1e-12
    assert np.allclose(E_loc[nonzero], ctx.E_0, atol=1e-6)


def test_sampling_matches_born():
    ctx = Hubbard(4, U=4.0)
    rng = np.random.default_rng(42)
    samples = ctx.sample(200_000, rng)
    ints = samples @ (1 << np.arange(2 * ctx.L, dtype=np.int64))
    idx = np.array([ctx.idx[int(i)] for i in ints])
    p_emp = np.bincount(idx, minlength=len(ctx.states)) / len(idx)
    p_true = ctx.psi_0 ** 2
    tv = 0.5 * np.abs(p_emp - p_true).sum()
    assert tv < 0.02, f"TV={tv:.3f} too large"


@pytest.mark.parametrize("L", [4, 6])
def test_signs_consistent(L):
    ctx = Hubbard(L)
    assert ctx.signs.shape == (len(ctx.states),)
    # Reconstruct psi = signs * |psi|; should equal psi_0 up to global sign.
    psi = ctx.signs * np.abs(ctx.psi_0)
    assert np.allclose(psi, ctx.psi_0, atol=1e-12) or \
           np.allclose(psi, -ctx.psi_0, atol=1e-12)


def test_model_sample_in_sector():
    ctx = Hubbard(4, U=4.0)
    torch.manual_seed(0)
    model = ARRNN(L=4, n_up=2, n_dn=2)
    samples = model.sample(256).cpu().numpy()
    n_up = samples[:, :4].sum(axis=1)
    n_dn = samples[:, 4:].sum(axis=1)
    assert (n_up == 2).all() and (n_dn == 2).all()
