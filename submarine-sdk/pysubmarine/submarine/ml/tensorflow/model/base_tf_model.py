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

import logging
from abc import ABC

import numpy as np
import tensorflow as tf

from submarine.ml.abstract_model import AbstractModel
from submarine.ml.tensorflow.parameters import default_parameters
from submarine.ml.tensorflow.registries import input_fn_registry
from submarine.utils.env import (get_from_dicts, get_from_json,
                                 get_from_registry)
from submarine.utils.tf_utils import get_tf_config

logger = logging.getLogger(__name__)


# pylint: disable=W0221
class BaseTFModel(AbstractModel, ABC):

    def __init__(self, model_params=None, json_path=None):
        super().__init__()
        self.model_params = get_from_dicts(model_params, default_parameters)
        self.model_params = get_from_json(json_path, self.model_params)
        self._sanity_checks()
        logging.info("Model parameters : %s", self.model_params)
        self.input_type = self.model_params['input']['type']
        self.model_dir = self.model_params['output']['save_model_dir']
        self.config = get_tf_config(self.model_params)
        self.model = tf.estimator.Estimator(model_fn=self.model_fn,
                                            model_dir=self.model_dir,
                                            params=self.model_params,
                                            config=self.config)

    def train(self, train_input_fn=None, eval_input_fn=None, **kwargs):
        """
        Trains a pre-defined tensorflow estimator model with given training data
        :param train_input_fn: A function that provides input data for training.
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: None
        """
        if train_input_fn is None:
            train_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                    filepath=self.model_params['input']['train_data'],
                    **self.model_params['training'])
        if eval_input_fn is None:
            eval_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                    filepath=self.model_params['input']['valid_data'],
                    **self.model_params['training'])

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec,
                                        **kwargs)

    def evaluate(self, eval_input_fn=None, **kwargs):
        """
        Evaluates a pre-defined Tensorflow estimator model with given evaluate data
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: A dict containing the evaluation metrics specified in `eval_input_fn` keyed by
        name, as well as an entry `global_step` which contains the value of the
        global step for which this evaluation was performed
        """
        if eval_input_fn is None:
            eval_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                    filepath=self.model_params['input']['valid_data'],
                    **self.model_params['training'])

        return self.model.evaluate(input_fn=eval_input_fn, **kwargs)

    def predict(self, predict_input_fn=None, **kwargs):
        """
        Yields predictions with given features.
        :param predict_input_fn: A function that constructs the features.
         Prediction continues until input_fn raises an end-of-input exception
        :return: Evaluated values of predictions tensors.
        """
        if predict_input_fn is None:
            predict_input_fn = get_from_registry(
                self.input_type, input_fn_registry)(
                    filepath=self.model_params['input']['test_data'],
                    **self.model_params['training'])

        return self.model.predict(input_fn=predict_input_fn, **kwargs)

    def _sanity_checks(self):
        assert 'input' in self.model_params, (
            'Does not define any input parameters')
        assert 'type' in self.model_params['input'], (
            'Does not define any input type')
        assert 'output' in self.model_params, (
            'Does not define any output parameters')

    def model_fn(self, features, labels, mode, params):
        seed = params["training"]["seed"]
        np.random.seed(seed)
        tf.set_random_seed(seed)
