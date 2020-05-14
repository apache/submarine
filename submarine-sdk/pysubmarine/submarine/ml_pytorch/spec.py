from .utils import read_file 

from .data import get_dataset_fn  
from .optimizer import get_optimizer_fn 
from .model import get_model_fn 
from .criterion import get_criterion_fn 
from .metric import get_metric_fn 

import torch 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 

from torch import distributed 

import json 

class Spec:
    def __init__(self, json_path): 
        self.spec = json.load(read_file(json_path)) 

    def get_dataloader_fn_train(self): 
        def _dataloader_fn(): 
            dataset = get_dataset_fn(key=self.spec['input']['type'])(path=self.spec['input']['train_data'])
            sampler = DistributedSampler(dataset)
            return DataLoader(
                dataset=dataset, 
                batch_size=self.spec['training']['batch_size'], 
                sampler=sampler,
                num_workers=self.spec['training']['num_threads'], 
                collate_fn=dataset.collate_fn 
            )
        return _dataloader_fn 

    def get_dataloader_fn_val(self): 
        def _dataloader_fn(): 
            dataset = get_dataset_fn(key=self.spec['input']['type'])(path=self.spec['input']['val_data'])
            sampler = DistributedSampler(dataset)
            return DataLoader(
                dataset=dataset, 
                batch_size=self.spec['training']['batch_size'], 
                sampler=sampler,
                num_workers=self.spec['training']['num_threads'], 
                collate_fn=dataset.collate_fn  
            )
        return _dataloader_fn 

    def get_dataloader_fn_test(self): 
        def _dataloader_fn(): 
            dataset = get_dataset_fn(key=self.spec['input']['type'])(path=self.spec['input']['test_data'])
            sampler = DistributedSampler(dataset)
            return DataLoader(
                dataset=dataset, 
                batch_size=self.spec['training']['batch_size'], 
                sampler=sampler,
                num_workers=self.spec['training']['num_threads'], 
                collate_fn=dataset.collate_fn  
            )
        return _dataloader_fn 

    def get_optimizer_fn(self): 
        def _optimizer_fn(params): 
            return get_optimizer_fn(key=self.spec['optimizer']['name'])(
                params=params, 
                **self.spec['optimizer']['kwargs']
            )
        return _optimizer_fn 

    def get_model_fn(self):
        def _model_fn(): 
            return get_model_fn(key=self.spec['model']['name'])(
                **self.spec['model']['kwargs']
            ) 
        return _model_fn 

    def get_criterion_fn(self): 
        def _criterion_fn(): 
            return get_criterion_fn(key=self.spec['criterion']['name'])(
                **self.spec['criterion']['kwargs']
            )
        return _criterion_fn 

    def get_metric_fn(self): 
        def _metric_fn(): 
            return get_metric_fn(key=self.spec['output']['metric'])  
        return _metric_fn 

    def is_distributed(self): 
        return self.spec['training']['mode'] == 'distributed' 

    def num_epochs(self): 
        return self.spec['training']['num_epochs'] 

    def batch_size(self): 
        return self.spec['training']['batch_size'] 

    def device(self): 
        if self.spec['training']['num_gpus'] > 0: 
            return torch.device('cuda:0') 
        else: 
            return torch.device('cpu') 

    def seed(self): 
        return self.spec['training']['seed'] 

    def checkpoint_dir(self): 
        return self.spec['output']['save_model_dir'] 

