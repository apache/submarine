from submarine.ml.pytorch.model.abstract_model import AbstractModel 

from submarine.ml.pytorch.parameters import default_parameters 

from submarine.ml.pytorch.registries import input_fn_registry 

from submarine.utils.env import get_from_registry
from submarine.utils.env import get_from_dicts
from submarine.utils.env import get_from_json 


from submarine.ml.pytorch.data.utils import write_file 

import torch 
from torch import distributed 
from torch.nn.parallel import DistributedDataParallel 

import os 
import logging 

logger = logging.getLogger(__name__) 


class BasePytorchModel(AbstractModel): 
    def __init__(self, model_params=None, json_path=None): 
        super().__init__() 
        self.model_params = get_from_json(json_path, get_from_dicts(model_params, default_parameters))
        self._sanity_checks() 
        logging.info('Model parameters: %s', self.model_params) 



    def train(self, train_input_fn=None, eval_input_fn=None, **kwargs): 
        self.init_training() 

        if train_input_fn is None: 
            train_input_fn = get_from_registry(self.model_params['input']['type'], input_fn_registry)(
                filepath=self.model_params['input']['train_data'][0],
                **self.model_params['training']
            )
        if eval_input_fn is None:
            eval_input_fn = get_from_registry(self.model_params['input']['type'], input_fn_registry)(
                filepath=self.model_params['input']['valid_data'][0],
                **self.model_params['training']
            )

        trainloader = train_input_fn() 
        valloader = eval_input_fn()

        for epoch in range(self.model_params['training']['num_epochs']): 
            for data in trainloader:
                 
        
        
        return 

    def evaluate(self): 
        return 

    def predict(self): 
        return 

    def init_training(self): 
        if self.model_params['training']['mode'] == 'distributed':  
            distributed.init_process_group(
                backend=distributed.Backend.GLOO, 
                init_method=os.environ.get('INIT_METHOD', 'tcp://127.0.0.1:23456'), 
                rank=int(os.environ.get('RANK')), 
                world_size=int(os.environ.get('WORLD'))
            )
        torch.manual_seed(self.model_params['training']['seed'])


    def _sanity_checks(self):
        assert 'input' in self.model_params, (
            'Does not define any input parameters'
        )
        assert 'type' in self.model_params['input'], (
            'Does not define any input type'
        )
        assert 'output' in self.model_params, (
            'Does not define any output parameters'
        )

    def save_checkpoint(self): 
        write_file()
        






