from .libsvm_dataset import LIBSVMDataset 

LIBSVM = 'libsvm' 


dataset_fn = {
    LIBSVM: LIBSVMDataset 
}

def get_dataset_fn(key):
    return dataset_fn[key] 



__all__ = ['get_dataset_fn'] 