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

from submarine.utils.tf_utils import get_tf_config


@pytest.mark.skipif(tf.__version__ >= "2.0.0", reason="requires tf1")
def test_get_tf_config():
    params = {"training": {"mode": "test"}}
    with pytest.raises(ValueError, match="mode should be local or distributed"):
        get_tf_config(params)

    # conf for local training
    params.update(
        {
            "training": {"mode": "local", "log_steps": 10},
            "resource": {"num_cpu": 4, "num_thread": 4, "num_gpu": 1},
        }
    )
    get_tf_config(params)

    # conf for distributed training
    params.update(
        {
            "training": {"mode": "distributed", "log_steps": 10},
            "resource": {"num_cpu": 4, "num_thread": 4, "num_gpu": 2},
        }
    )
    get_tf_config(params)
