def mse(y_test, y_pred):
    mse_sum = 0
    return ((y_test - y_pred) ** 2).sum() / len(y_test)
