"""Sample-diversity metrics — early warning for mode collapse in
generative models."""
from collections import Counter

import numpy as np


def unique_fraction(samples_bits):
    """Fraction of samples that are unique. 1.0 = all distinct. Tiny
    values flag a model collapsing onto a few modes."""
    rows = [tuple(row) for row in np.asarray(samples_bits, dtype=np.uint8)]
    return float(len(set(rows)) / len(rows))


def top1_fraction(samples_bits):
    """Fraction of samples that fall on the single most-sampled bitstring.
    Large values (>> 1/k) flag mode collapse / peaking."""
    rows = [tuple(row) for row in np.asarray(samples_bits, dtype=np.uint8)]
    cnt = Counter(rows)
    return float(cnt.most_common(1)[0][1] / len(rows))
