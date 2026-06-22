"""Sample-diversity metrics — flag mode collapse."""
from collections import Counter

import numpy as np


def unique_fraction(samples_bits):
    rows = [tuple(row) for row in np.asarray(samples_bits, dtype=np.uint8)]
    return float(len(set(rows)) / len(rows))


def top1_fraction(samples_bits):
    rows = [tuple(row) for row in np.asarray(samples_bits, dtype=np.uint8)]
    return float(Counter(rows).most_common(1)[0][1] / len(rows))
