from ..utils import read_file 
from ..utils import write_file 

import torch 
from torch import distributed 
from torch.nn.parallel import DistributedDataParallel 


import os 
import io 
from argparse import ArgumentParser 

class Trainer: 
    def __init__(self, spec):  
        self.spec = spec 
        
        self.init_process_group() 

        self.trainloader = spec.get_dataloader_fn_train()() 
        self.valloader = spec.get_dataloader_fn_val()() 
        self.testloader = spec.get_dataloader_fn_test()() 
        self.criterion = spec.get_criterion_fn()() 
        self.model = DistributedDataParallel(spec.get_model_fn()().to(spec.device())) 
        self.optimizer = spec.get_optimizer_fn()(params=self.model.parameters()) 
        self.metric = spec.get_metric_fn()()


    def init_process_group(self): 
        parser = ArgumentParser() 
        parser.add_argument('--rank', '-r', type=int, default=int(os.environ.get('RANK', 0))) 
        parser.add_argument('--world_size', '-w', type=int, default=int(os.environ.get('WORLD', 1))) 
        parser.add_argument('--init_method', '-i', type=str, default=os.environ.get('INIT_METHOD', 'tcp://127.0.0.1:23456')) 
        parser.add_argument('--backend', type=str, default=distributed.Backend.GLOO) 
        args, _ = parser.parse_known_args() 
        print(args)
        distributed.init_process_group(
            backend=args.backend, 
            init_method=args.init_method, 
            world_size=args.world_size, 
            rank=args.rank 
        )  

        torch.manual_seed(self.spec.seed()) 

    def __del__(self): 
        distributed.destroy_process_group() 
        

    def train(self): 
        for idx_batch, batch in enumerate(self.trainloader): 
            sample, target = batch 
            output = self.model(sample) 
            loss = self.criterion(output, target) 
            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

    def evaluate(self): 
        outputs = []
        targets = [] 
        with torch.no_grad(): 
            for ids_batch, batch in enumerate(self.valloader): 
                sample, target = batch 
                output = self.model(sample) 

                outputs.append(output) 
                targets.append(target) 

        return self.metric(
            torch.cat(targets, dim=0).cpu().numpy(), 
            torch.cat(outputs, dim=0).cpu().numpy() 
        )

    def predict(self): 
        outputs = []
        with torch.no_grad(): 
            for ids_batch, batch in enumerate(self.testloader): 
                sample, target = batch 
                output = self.model(sample) 

                outputs.append(output) 
        ### TODO: fix this 
        return torch.cat(outputs, dim=0).cpu().numpy()
        ###

    def fit(self): 
        ### TODO: fix this 
        best_eval_score = 0.0 
        ###
        for epoch in range(self.spec.num_epochs()): 
            self.trainloader.sampler.set_epoch(epoch) 
            self.train() 
            eval_score = self.evaluate() 
            ### TODO: fix this 
            if eval_score > best_eval_score: 
                best_eval_score = eval_score  
            ### 
                self.save_checkpoint()
        return best_eval_score 


    def save_checkpoint(self): 
        with io.BytesIO() as buffer:  
            torch.save(
                {
                    'model': self.model.module.state_dict(), 
                    'optimizer': self.optimizer.state_dict() 
                }, buffer 
            )
            write_file(
                buffer,  
                path=os.path.join(self.spec.checkpoint_dir(), 'ckpt.pkl')
            ) 
        

