from numpy import sqrt
def root_mean_square_error(targets, predictions):
    return sqrt(sum([pow(t-y, 2) for t, y in zip(targets,predictions)])/len(targets))

def mean_absolute_scale_error(y_true, y_pred):
    naive_error_fn = lambda targets, predictions: (
        sum([abs(t-y) for t,y in zip(targets, predictions)]) / len(targets)
    )
    naive_error = naive_error_fn(y_true[1:], y_true[:-1])
    
    mase = sum([abs(t-y)/naive_error for t,y in zip(y_true, y_pred)]) / len(y_true)
    return mase

def error_variance(y_true, y_pred):
    e_mean = sum([abs(t-y) for y,t in zip(y_pred, y_true)])/len(y_true)
    y_mean = sum(y_true)/len(y_true)
    error_variance = sum([pow((abs(y_hat - y) / y_mean) - e_mean, 2) for y_hat,y in zip(y_pred, y_true)]) / len(y_true)
    return error_variance