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

import os

import pytest

LIBSVM_DATA = """1 1:0 2:0.051495 3:0.5 4:0.1 5:0.113437 6:0.874 7:0.01 8:0.08 9:0.028 10:0
1 1:1.35 2:0.031561 3:0.45 4:0.56 5:0.000031 6:0.056 7:0.27 8:0.58 9:0.056 10:0.166667
1 1:0.05 2:0.004983 3:0.19 4:0.14 5:0.000016 6:0.006 7:0.01 8:0.14 9:0.014 10:0.166667
1 1:0.2 2:0.004983 3:0 4:0.12 5:0.016422 6:0.268 7:0.04 8:0.7 9:0.144 10:0.166667
1 1:0 2:0.051495 3:0.5 4:0.1 5:0.113437 6:0.874 7:0.01 8:0.08 9:0.028 10:0
1 1:1.35 2:0.031561 3:0.45 4:0.56 5:0.000031 6:0.056 7:0.27 8:0.58 9:0.056 10:0.166667
1 1:0.05 2:0.004983 3:0.19 4:0.14 5:0.000016 6:0.006 7:0.01 8:0.14 9:0.014 10:0.166667
1 1:0.2 2:0.004983 3:0 4:0.12 5:0.016422 6:0.268 7:0.04 8:0.7 9:0.144 10:0.166667
"""


@pytest.fixture
def get_model_param(tmpdir):
    data_file = os.path.join(str(tmpdir), "libsvm.txt")
    save_model_dir = os.path.join(str(tmpdir), "experiment")
    with open(data_file, "wt") as writer:
        writer.write(LIBSVM_DATA)

    params = {
        "input": {
            "train_data": data_file,
            "valid_data": data_file,
            "test_data": data_file,
            "type": "libsvm"
        },
        "output": {
            "save_model_dir": save_model_dir,
            "metric": "auc"
        },
        "training": {
            "batch_size": 256,
            "num_epochs": 1,
            "field_size": 10,
            "feature_size": 1000
        }
    }

    yield params
    os.remove(data_file)
