import math


def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mse(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def fit_sincos(values, period=24):
    """Least-squares fit: y ≈ a_sin*sin(2π*t/T) + a_cos*cos(2π*t/T)."""
    n = len(values)
    sins = [math.sin(2 * math.pi * t / period) for t in range(n)]
    coss = [math.cos(2 * math.pi * t / period) for t in range(n)]
    ss = sum(s * s for s in sins)
    sc = sum(s * c for s, c in zip(sins, coss))
    cc = sum(c * c for c in coss)
    sy = sum(s * v for s, v in zip(sins, values))
    cy = sum(c * v for c, v in zip(coss, values))
    det = ss * cc - sc * sc
    if abs(det) < 1e-10:
        return 0.0, 0.0
    return (cc * sy - sc * cy) / det, (ss * cy - sc * sy) / det
