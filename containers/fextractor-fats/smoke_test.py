from __future__ import print_function

import json
import math
import sys

import numpy as np
import FATS


def build_light_curve(n=256):
    # Irregular, strictly increasing observing times.
    dt = 0.8 + 0.25 * (1.0 + np.sin(np.arange(n, dtype=float) * 0.37))
    time = np.cumsum(dt)

    # Two smooth, non-degenerate synthetic photometric bands.
    magnitude = (
        15.0
        + 0.35 * np.sin(2.0 * np.pi * time / 17.3)
        + 0.08 * np.sin(2.0 * np.pi * time / 4.7)
    )
    magnitude2 = (
        15.6
        + 0.29 * np.sin(2.0 * np.pi * time / 17.3 + 0.22)
        + 0.06 * np.sin(2.0 * np.pi * time / 4.7 + 0.11)
    )

    error = 0.03 + 0.005 * (1.0 + np.sin(np.arange(n, dtype=float) * 0.19))
    error2 = 0.04 + 0.005 * (1.0 + np.cos(np.arange(n, dtype=float) * 0.23))

    # FATS expects this nine-vector layout when Data='all':
    # magnitude, time, error, magnitude2,
    # aligned_magnitude, aligned_magnitude2, aligned_time,
    # aligned_error, aligned_error2.
    # Our synthetic bands are already sampled at the same times, so aligned
    # vectors are identical to the original ones.
    return np.asarray([
        magnitude,
        time,
        error,
        magnitude2,
        magnitude,
        magnitude2,
        time,
        error,
        error2,
    ])


def main():
    data = build_light_curve()

    fs = FATS.FeatureSpace(Data='all')
    fs.calculateFeature(data)

    names = list(fs.result(method='features'))
    values = list(fs.result(method='array'))

    if not names:
        raise RuntimeError('FATS returned no features')
    if len(names) != len(values):
        raise RuntimeError(
            'Feature name/value length mismatch: %d != %d' %
            (len(names), len(values))
        )

    # Verify every result can be represented as a scalar float. NaN/Inf are
    # allowed here because individual legacy FATS statistics can be undefined
    # for particular inputs; the integration layer can choose its policy.
    converted = []
    for name, value in zip(names, values):
        arr = np.asarray(value)
        if arr.size != 1:
            raise RuntimeError(
                'Feature %s returned non-scalar shape %r' % (name, arr.shape)
            )
        converted.append(float(arr.reshape(-1)[0]))

    print('FATS import OK')
    print('FATS feature count: %d' % len(names))
    print('FATS feature names: %s' % ','.join(names))
    print('FATS vector shape: (%d,)' % len(converted))

    # Machine-readable summary useful in CI/container inspection.
    print(json.dumps({
        'feature_count': len(names),
        'feature_names': names,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
