from submarine.ml.pytorch.data.input import libsvm_input_fn 

LIBSVM = 'libsvm' 

input_fn_registry = {
    LIBSVM: libsvm_input_fn 
}

def get_input_fn(key): 
    return input_fn_registry[key] 


