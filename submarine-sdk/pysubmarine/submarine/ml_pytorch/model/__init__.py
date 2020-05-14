from . import ctr 

from torch import nn 

def get_model_fn(key): 
    module_name, model_name = key.split('.') 
    def _condition_fn(x): 
        k, v = x 
        return isinstance(v, type) and issubclass(v, nn.Module) and (k.lower() == model_name.lower()) 
    _, model_fn = next(iter(filter(_condition_fn, vars(globals()[module_name.lower()]).items())))
    return model_fn  


__all__ = ['get_model_fn']
