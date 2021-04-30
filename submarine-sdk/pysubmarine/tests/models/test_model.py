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

from submarine import ModelsClient
from pytorch import LinearNNModel
import numpy as np
import pytest

# Temporarily skip these tests after the following is solved:
# TODO: Setup cluster by helm in CI/CD to enable mlflow server connection
# TODO: Set an cooldown time between each test case
@pytest.mark.skip(reason="no way of currently testing this")
class TestSubmarineModelsClient():
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_log_model(self):
        client = ModelsClient()
        model = LinearNNModel()
        name = "simple-nn-model"
        client.log_model(name, model)

    def test_update_model(self):
        client = ModelsClient()
        name = "simple-nn-model"
        new_name = "new-simple-nn-model"
        client.update_model(name, new_name)

    def test_load_model(self):
        client = ModelsClient()
        name = "simple-nn-model"
        version = "1"
        model = client.load_model(name, version)
        x = np.float32([[1.0], [2.0]])
        y = model.predict(x)
        assert y.shape[0] == 2
        assert y.shape[1] == 1

    def test_delete_model(self):
        client = ModelsClient()
        name = "simple-nn-model"
        client.delete_model(name, '1')
