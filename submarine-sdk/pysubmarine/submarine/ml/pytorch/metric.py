from sklearn import metrics
from types import FunctionType


def get_metric_fn(key):
    def _condition_fn(x):
        k, v = x
        return isinstance(v, FunctionType) and (k.lower() == key.lower())
    _, metric_fn = next(iter(filter(_condition_fn, vars(metrics).items())))
    return metric_fn
