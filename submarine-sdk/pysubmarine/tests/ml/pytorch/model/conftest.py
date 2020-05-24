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
import os

# noqa
LIBSVM_DATA = """
0 0:0 1:0 2:0 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0 11:0 12:0 13:0 14:0 15:24 16:38 17:0 18:0 19:60 20:0 21:0 22:33 23:74 24:29 25:78 26:0 27:84 28:36 29:0 30:0 31:0 32:0 33:31 34:0 35:0 36:41 37:0 38:22
0 0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:0 8:1 9:0 10:1 11:0 12:1 13:0 14:1 15:0 16:0 17:0 18:1 19:60 20:1 21:0 22:33 23:74 24:0 25:78 26:1 27:0 28:0 29:1 30:1 31:0 32:1 33:0 34:0 35:0 36:0 37:0 38:0
0 0:1 1:1 2:2 3:2 4:2 5:2 6:2 7:0 8:2 9:0 10:2 11:1 12:2 13:1 14:2 15:1 16:1 17:0 18:0 19:60 20:1 21:0 22:0 23:74 24:1 25:78 26:2 27:84 28:1 29:2 30:2 31:1 32:2 33:1 34:1 35:0 36:1 37:1 38:1 
0 0:2 1:2 2:3 3:3 4:3 5:3 6:3 7:1 8:3 9:1 10:3 11:0 12:3 13:0 14:3 15:24 16:38 17:0 18:1 19:60 20:1 21:0 22:1 23:0 24:29 25:0 26:2 27:1 28:36 29:3 30:3 31:1 32:2 33:31 34:0 35:0 36:2 37:1 38:1
0 0:3 1:3 2:3 3:0 4:4 5:4 6:2 7:1 8:3 9:0 10:1 11:0 12:4 13:2 14:4 15:24 16:2 17:0 18:2 19:60 20:1 21:0 22:33 23:74 24:2 25:78 26:0 27:84 28:2 29:3 30:93 31:1 32:2 33:2 34:0 35:1 36:3 37:1 38:1
0 0:2 1:3 2:3 3:3 4:5 5:3 6:3 7:1 8:4 9:1 10:3 11:0 12:3 13:3 14:76 15:24 16:38 17:1 18:3 19:0 20:1 21:0 22:0 23:1 24:29 25:1 26:2 27:84 28:36 29:4 30:93 31:1 32:2 33:31 34:2 35:2 36:41 37:1 38:1
0 0:2 1:0 2:4 3:3 4:6 5:3 6:3 7:2 8:5 9:1 10:3 11:0 12:3 13:20 14:76 15:24 16:38 17:1 18:1 19:60 20:1 21:0 22:0 23:74 24:29 25:78 26:2 27:84 28:36 29:4 30:93 31:1 32:2 33:31 34:0 35:3 36:4 37:1 38:1
1 0:0 1:4 2:4 3:0 4:7 5:4 6:4 7:1 8:3 9:0 10:1 11:0 12:4 13:0 14:3 15:24 16:38 17:2 18:2 19:1 20:0 21:0 22:2 23:74 24:29 25:2 26:2 27:1 28:36 29:0 30:3 31:1 32:2 33:31 34:0 35:1 36:2 37:1 38:1
0 0:2 1:5 2:5 3:4 4:8 5:5 6:5 7:3 8:6 9:1 10:1 11:0 12:5 13:3 14:5 15:2 16:1 17:0 18:0 19:60 20:1 21:0 22:33 23:74 24:3 25:78 26:1 27:84 28:3 29:0 30:93 31:1 32:2 33:3 34:0 35:1 36:1 37:1 38:1
0 0:2 1:6 2:3 3:1 4:9 5:6 6:4 7:0 8:1 9:1 10:1 11:0 12:6 13:3 14:76 15:24 16:38 17:0 18:4 19:2 20:1 21:0 22:33 23:0 24:29 25:0 26:2 27:84 28:36 29:5 30:93 31:1 32:2 33:31 34:0 35:1 36:41 37:1 38:1
"""  # noqa


@pytest.fixture
def get_model_param(tmpdir):
    data_file = os.path.join(str(tmpdir), "libsvm.txt")
    save_model_dir = os.path.join(str(tmpdir), "experiment")
    os.mkdir(save_model_dir)

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
            "metric": "roc_auc_score"
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 1,
            "log_steps": 10,
            "num_threads": 0,
            "num_gpus": 0,
            "seed": 42,
            "mode": "distributed",
            "backend": "gloo"
        },
        "model": {
            "name": "ctr.deepfm",
            "kwargs": {
                "field_dims":
                [15, 52, 30, 19, 111, 51, 26, 19, 53, 5, 13, 8, 23,
                    21, 77, 25, 39, 11, 8, 61, 15, 3, 34, 75, 30, 79, 11, 85, 37,
                    10, 94, 19, 5, 32, 6, 12, 42, 18, 23],
                "out_features": 1,
                "embedding_dim": 16,
                "hidden_units": [400, 400],
                "dropout_rates": [0.2, 0.2]
            }
        },
        "loss": {
            "name": "BCEWithLogitsLoss",
            "kwargs": {}
        },
        "optimizer": {
            "name": "adam",
            "kwargs": {
                "lr": 1e-3
            }
        },
        "resource": {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_threads": 0
        }
    }

    yield params
    os.remove(data_file)
