import numpy as np

def log_sum_exp(x, axis=-1):
    a = x.max(axis=axis, keepdims=True)
    out = a + np.log(np.sum(np.exp(x - a), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def kl_normal(qm, qv, pm, pv):
    (qm, qv), (pm, pv) = q, p
    return 0.5 * np.sum(np.log(pv) - np.log(qv) + qv/pv + np.square(qm - pm) / pv - 1, axis=-1)

def convert_to_ssl(x, y, n_labels, n_classes):
    if y.shape[-1] == n_classes:
        y_sparse = y.argmax(1)
    else:
        y_sparse = y
    x_label, y_label = [], []
    for i in xrange(n_classes):
        idx = y_sparse == i
        x_cand, y_cand = x[idx], y[idx]
        idx = np.random.choice(len(x_cand), n_labels/n_classes, replace=False)
        x_select, y_select = x_cand[idx], y_cand[idx]
        x_label += [x_select]
        y_label += [y_select]
    x_label = np.array(x_label).reshape(-1, *x.shape[1:])
    y_label = np.array(y_label).reshape(-1, *y.shape[1:])
    return x_label, y_label
