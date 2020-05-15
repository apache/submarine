from abc import ABC
import logging

from submarine.ml.pytorch.metric import get_metric_fn
from submarine.ml.pytorch.registries import input_fn_registry
from submarine.utils.fileio import write_file
from submarine.ml.abstract_model import AbstractModel
import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel

import os
import io
from argparse import ArgumentParser
from submarine.ml.pytorch.optimizer import get_optimizer
from submarine.ml.pytorch.loss import get_loss_fn
from submarine.ml.tensorflow.parameters import default_parameters
from submarine.utils.env import get_from_registry, get_from_json, get_from_dicts
from submarine.utils.pytorch_utils import get_device

logger = logging.getLogger(__name__)


# pylint: disable=W0221
class BasePyTorchModel(AbstractModel, ABC):
    def __init__(self, params=None, json_path=None):
        super().__init__()
        self.params = get_from_dicts(params, default_parameters)
        self.params = get_from_json(json_path, self.params)
        self.input_type = self.params['input']['type']
        logging.info("Model parameters : %s", self.params)

        self.init_process_group()
        self.model = DistributedDataParallel(self.model_fn(self.params).to(get_device(self.params)))
        self.optimizer = get_optimizer(key=self.params['optimizer']['name'])(
                params=self.model.parameters(),
                **self.params['optimizer']['kwargs']
            )
        self.loss = get_loss_fn(key=self.params['loss']['name'])(
            **self.params['loss']['kwargs'])
        self.metric = get_metric_fn(key=self.params['output']['metric'])

    def init_process_group(self):
        parser = ArgumentParser()
        parser.add_argument('--rank', '-r', type=int, default=int(os.environ.get('RANK', 0)))
        parser.add_argument('--world_size', '-w', type=int, default=int(os.environ.get('WORLD', 1)))
        parser.add_argument('--init_method', '-i', type=str,
                            default=os.environ.get('INIT_METHOD', 'tcp://127.0.0.1:23456'))
        parser.add_argument('--backend', type=str, default=distributed.Backend.GLOO)
        args, _ = parser.parse_known_args()
        print(args)
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank
        )

    def __del__(self):
        distributed.destroy_process_group()

    def train(self, train_loader):
        for _, batch in enumerate(train_loader):
            sample, target = batch
            output = self.model(sample)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()

    def evaluate(self):
        outputs = []
        targets = []

        valid_loader = get_from_registry(
            self.input_type, input_fn_registry)(
            filepath=self.params['input']['valid_data'],
            **self.params['training'])()

        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
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

        test_loader = get_from_registry(
            self.input_type, input_fn_registry)(
            filepath=self.params['input']['test_data'],
            **self.params['training'])()

        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                sample, _ = batch
                output = self.model(sample)
                outputs.append(output)
        # TODO: fix this
        return torch.cat(outputs, dim=0).cpu().numpy()

    def fit(self):
        # TODO: fix this
        best_eval_score = 0.0
        train_loader = get_from_registry(
            self.input_type, input_fn_registry)(
            filepath=self.params['input']['train_data'],
            **self.params['training'])()

        for epoch in range(self.params['training']['num_epochs']):
            train_loader.sampler.set_epoch(epoch)
            self.train(train_loader)
            eval_score = self.evaluate()
            # TODO: fix this
            if eval_score > best_eval_score:
                best_eval_score = eval_score
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
                path=os.path.join(self.params['output']['save_model_dir'], 'ckpt.pkl')
            )

    def model_fn(self, params):
        seed = params["training"]["seed"]
        torch.manual_seed(seed)
