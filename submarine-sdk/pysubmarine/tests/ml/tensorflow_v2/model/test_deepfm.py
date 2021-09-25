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
import tensorflow as tf

from submarine.ml.tensorflow_v2.model import DeepFM


@pytest.mark.skipif(tf.__version__ < "2.0.0", reason="requires tf2")
def test_run_deepfm(get_model_param):
    params = get_model_param

    model = DeepFM(model_params=params)
    model.train()
    model.evaluate()
    model.predict()
