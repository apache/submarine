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

default_parameters = {
    "output": {
        "save_model_dir": "./output",
        "metric": "roc_auc"
    },
    "training": {
        "batch_size": 64,
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
            "out_features": 1,
            "embedding_dim": 256,
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
        "num_cpus": 4,
        "num_gpus": 0,
        "num_threads": 0
    }
}
