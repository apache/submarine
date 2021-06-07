"""
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
"""

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from keras import LinearNNModelKeras
from pytorch import LinearNNModelTorch
from submarine import ModelsClient
from submarine.models.client import types


class TestSubmarineModelsClient():

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_log_model_pytorch(self, mocker):
        mock_method = mocker.patch.object(ModelsClient, "log_model")
        client = ModelsClient("pytorch")
        model = LinearNNModelTorch()
        name = "simple-nn-model"
        client.log_model(name, model)
        mock_method.assert_called_once_with("simple-nn-model", model)

    def test_log_model_keras(self, mocker):
        mock_method = mocker.patch.object(ModelsClient, "log_model")
        client = ModelsClient("keras")
        model = LinearNNModelKeras()
        name = "simple-nn-model"
        client.log_model(name, model)
        mock_method.assert_called_once_with("simple-nn-model", model)   

    def test_log_model_sklearn(self, mocker):
        mock_method = mocker.patch.object(ModelsClient, "log_model")
        client = ModelsClient("sklearn")
        model = LogisticRegression()
        name = "simple-nn-model"
        client.log_model(name, model)
        mock_method.assert_called_once_with("simple-nn-model", model) 

    def test_update_model(self, mocker):
        mock_method = mocker.patch.object(MlflowClient,
                                          "rename_registered_model")
        client = ModelsClient()
        name = "simple-nn-model"
        new_name = "new-simple-nn-model"
        client.update_model(name, new_name)
        mock_method.assert_called_once_with(name="simple-nn-model",
                                            new_name="new-simple-nn-model")

    def test_load_model(self, mocker):
        mock_method = mocker.patch.object(mlflow.pyfunc, "load_model")
        mock_method.return_value = mlflow.pytorch._PyTorchWrapper(
            LinearNNModelTorch())
        client = ModelsClient()
        name = "simple-nn-model"
        version = "1"
        model = client.load_model(name, version)
        mock_method.assert_called_once_with(
            model_uri="models:/simple-nn-model/1")
        x = np.float32([[1.0], [2.0]])
        y = model.predict(x)
        assert y.shape[0] == 2
        assert y.shape[1] == 1

    def test_delete_model(self, mocker):
        mock_method = mocker.patch.object(MlflowClient, "delete_model_version")
        client = ModelsClient()
        name = "simple-nn-model"
        client.delete_model(name, '1')
        mock_method.assert_called_once_with(name="simple-nn-model", version="1")
