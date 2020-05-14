from torch import optim 

def get_optimizer_fn(key): 
    def _condition_fn(x): 
        k, v = x 
        return isinstance(v, type) and issubclass(v, optim.Optimizer) and (k.lower() == key.lower()) 
    _, optimizer_fn = next(iter(filter(_condition_fn, vars(optim).items())))
    return optimizer_fn 




