def mse(y_test, y_pred):
    return (y_test - y_pred).norm(dim=1) ** 2 / y_test.norm(dim=1) ** 2
