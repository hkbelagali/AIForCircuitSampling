"""XEB estimators for Stage 2.

For RCS samples x_i drawn from a mixture
    q_F(x) = F * p_C(x) + (1 - F) / N
(noiseless ideal: F = 1; depolarized noise: 0 <= F <= 1), we want estimators
of F from k samples that beat the empirical linear estimator's
Var ~ Theta(1/k) constant.

All estimators here return a scalar F-hat. Under Porter-Thomas the population
expectations are calibrated so F-hat -> 1 for ideal samples and F-hat -> 0
for uniform-random samples. Variances differ.
"""

import numpy as np
from scipy.optimize import minimize_scalar


_EULER_GAMMA = 0.5772156649015329


def linear_xeb(samples_int, p_C):
    """Linear XEB:  F_lin = N * mean(p_C(x_i)) - 1."""
    if len(samples_int) == 0:
        return float("nan")
    N = int(len(p_C))
    return float(N * p_C[samples_int].mean() - 1.0)


def log_xeb(samples_int, p_C, eps=1e-300):
    """Logarithmic XEB:  F_log = mean(log(N * p_C(x_i))) + gamma_E.

    Under Porter-Thomas, E_{x~p_C}[log(N*p_C)] = 1 - gamma_E, so the
    estimator is centered at F=1 for ideal noiseless samples. Has lower
    variance than F_lin in the high-fidelity, deep-circuit regime.
    """
    if len(samples_int) == 0:
        return float("nan")
    N = int(len(p_C))
    vals = np.log(np.maximum(N * p_C[samples_int], eps))
    return float(vals.mean() + _EULER_GAMMA)


def truncated_linear_xeb(samples_int, p_C, q_low=0.0, q_high=0.99):
    """Linear XEB with empirical winsorization at quantiles (q_low, q_high)
    of the *sampled* p_C values. Trades a small bias for variance reduction
    in the PT heavy tail.
    """
    if len(samples_int) == 0:
        return float("nan")
    N = int(len(p_C))
    vals = p_C[samples_int]
    lo = float(np.quantile(vals, q_low)) if q_low > 0 else float(vals.min())
    hi = float(np.quantile(vals, q_high)) if q_high < 1 else float(vals.max())
    clipped = np.clip(vals, lo, hi)
    return float(N * clipped.mean() - 1.0)


def mle_fidelity_xeb(samples_int, p_C, fmin=0.0, fmax=1.5):
    """Maximum-likelihood fidelity estimate under the mixture model
        q_F(x) = F * p_C(x) + (1 - F) / N
    via 1-D minimization of the per-sample negative log-likelihood. F is not
    constrained to [0, 1] here so the estimator is unbiased for F=1 with
    finite sample noise; pin to [fmin, fmax] only to avoid pathological optima.
    """
    if len(samples_int) == 0:
        return float("nan")
    N = int(len(p_C))
    inv_N = 1.0 / N
    p_x = p_C[samples_int]

    def negloglik(F):
        q = F * p_x + (1.0 - F) * inv_N
        # If F is large and some samples have p_C < 1/N, q can be tiny but >0
        return float(-np.log(np.maximum(q, 1e-300)).mean())

    res = minimize_scalar(negloglik, bounds=(fmin, fmax), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)
