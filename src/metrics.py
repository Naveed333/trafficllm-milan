def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mse(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
