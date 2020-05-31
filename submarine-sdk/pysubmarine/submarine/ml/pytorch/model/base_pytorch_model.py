# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from submarine.ml.pytorch.optimizer import get_optimizer
from submarine.ml.pytorch.loss import get_loss_fn
from submarine.ml.pytorch.parameters import default_parameters
from submarine.utils.env import get_from_registry, get_from_json, get_from_dicts
from submarine.utils.pytorch_utils import get_device

logger = logging.getLogger(__name__)


# pylint: disable=W0221
class BasePyTorchModel(AbstractModel, ABC):
    def __init__(self, params=None, json_path=None):
        super().__init__()
        self.params = get_from_dicts(params, default_parameters)
        self.params = get_from_json(json_path, self.params)
        self._sanity_check()
        logging.info("Model parameters : %s", self.params)
        self.input_type = self.params['input']['type']

        self.init_process_group()
        self.model = DistributedDataParallel(
            self.model_fn(self.params).to(get_device(self.params)))
        self.optimizer = get_optimizer(key=self.params['optimizer']['name'])(
            params=self.model.parameters(),
            **self.params['optimizer']['kwargs']
        )
        self.loss = get_loss_fn(key=self.params['loss']['name'])(
            **self.params['loss']['kwargs'])
        self.metric = get_metric_fn(key=self.params['output']['metric'])

    def init_process_group(self):
        distributed.init_process_group(
            backend=os.environ.get('backend', distributed.Backend.GLOO),
            init_method=os.environ.get('INIT_METHOD', 'tcp://127.0.0.1:23456'),
            world_size=int(os.environ.get('WORLD', 1)),
            rank=int(os.environ.get('RANK', 0))
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

        return torch.cat(outputs, dim=0).cpu().numpy()

    def fit(self):
        # TODO (andrewhsiehth):
        # Handle the comparison of different kinds of evaluation metrics.
        # E.g. For roc_auc, the higher the better.
        # But for mse, the lower the better.
        # The line "if eval_score > best_eval_score:"
        # should be replaced by a indicator function.
        best_eval_score = 0.0
        train_loader = get_from_registry(
            self.input_type, input_fn_registry)(
            filepath=self.params['input']['train_data'],
            **self.params['training'])()

        for epoch in range(self.params['training']['num_epochs']):
            train_loader.sampler.set_epoch(epoch)
            self.train(train_loader)
            eval_score = self.evaluate()

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
                path=os.path.join(
                    self.params['output']['save_model_dir'], 'ckpt.pkl')
            )

    def model_fn(self, params):
        seed = params["training"]["seed"]
        torch.manual_seed(seed)

    def _sanity_check(self):
        assert 'input' in self.params, (
            'Does not define any input parameters'
        )
        assert 'type' in self.params['input'], (
            'Does not define any input type'
        )
        assert 'output' in self.params, (
            'Does not define any output parameters'
        )
