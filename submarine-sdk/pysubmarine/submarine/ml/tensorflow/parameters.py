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
        "save_model_dir": "./experiment",
        "metric": "auc"
    },
    "training": {
        "batch_size": 512,
        "field_size": 39,
        "num_epochs": 3,
        "feature_size": 117581,
        "embedding_size": 256,
        "learning_rate": 0.0005,
        "batch_norm_decay": 0.9,
        "l2_reg": 0.0001,
        "deep_layers": [400, 400, 400],
        "dropout": [0.3, 0.3, 0.3],
        "batch_norm": "false",
        "optimizer": "adam",
        "log_steps": 10,
        "num_threads": 4,
        "num_gpu": 0,
        "seed": 77,
        "mode": "local"
    },
    "resource": {
        "num_cpu": 4,
        "num_gpu": 0,
        "num_thread": 0  # tf determines automatically
    }
}
