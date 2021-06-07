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

import os

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression
from keras import LinearNNModelKeras
from pytorch import LinearNNModelTorch
from submarine import ModelsClient, types
from submarine.models import constant


@pytest.fixture(name="models_client", scope="class", params=["pytorch", "keras", "sklearn"])
def models_client_fixture(request):
    client = ModelsClient("http://localhost:5001", "http://localhost:9000", request.param)
    return client

@pytest.fixture(name="models_client_pytorch", scope="class")
def models_client_pytorch_fixture():
    """
    Temporarily,can be removed when finished expansion of other funtions 
    """
    client = ModelsClient("http://localhost:5001", "http://localhost:9000")
    return client


@pytest.mark.e2e
class TestSubmarineModelsClientE2E():

    def test_model(self, models_client, models_client_pytorch):
        if models_client.type.name == types.PYTORCH.name:
            model = LinearNNModelTorch()

        elif models_client.type.name == types.KERAS.name:
            model = LinearNNModelKeras()

        elif models_client.type.name == types.SKLEARN.name:
            model = LogisticRegression()

        # log
        name = "simple-nn-model"
        models_client_pytorch.log_model(name, model)
        # update
        new_name = "new-simple-nn-model"
        models_client_pytorch.update_model(name, new_name)
        # load
        name = new_name
        version = "1"
        model = models_client_pytorch.load_model(name, version)
        x = np.float32([[1.0], [2.0]])
        y = model.predict(x)
        assert y.shape[0] == 2
        assert y.shape[1] == 1
        # delete
        models_client_pytorch.delete_model(name, '1')
