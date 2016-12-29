import numpy as np


def log_sum_exp(x, axis=-1):
    a = x.max(axis=axis, keepdims=True)
    out = a + np.log(np.sum(np.exp(x - a), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def kl_normal(qm, qv, pm, pv):
    (qm, qv), (pm, pv) = q, p
    return 0.5 * np.sum(np.log(pv) - np.log(qv) + qv/pv + np.square(qm - pm) / pv - 1, axis=-1)
