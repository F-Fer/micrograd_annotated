from my_micrograd.nn import Value

def mean_squared_error(y_pred, y):
    # Check if inputs are scalar
    if not hasattr(y_pred, '__iter__') and not hasattr(y, '__iter__'):
        return (y_pred - y) ** 2  # Direct computation for scalars

    return sum((y_out - y_true)**2 for y_out, y_true in zip(y_pred, y))

