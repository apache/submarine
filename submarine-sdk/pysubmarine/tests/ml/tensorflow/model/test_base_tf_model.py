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

import pytest

from submarine.ml.tensorflow.model.base_tf_model import BaseTFModel


def test_create_base_tf_model():
    params = {"learning rate": 0.05}
    with pytest.raises(AssertionError, match="Does not define any input parameters"):
        BaseTFModel(params)

    params.update({"input": {"train_data": "/tmp/train.csv"}})
    with pytest.raises(AssertionError, match="Does not define any input type"):
        BaseTFModel(params)

    params.update({"input": {"type": "libsvm"}})
    BaseTFModel(params)
