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

import argparse

from submarine.ml.tensorflow.model import DeepFM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", help="a JSON configuration file for DeepFM", type=str)
    parser.add_argument(
        "-task_type", default="train", help="train or evaluate, by default is train"
    )
    args = parser.parse_args()
    json_path = args.conf
    task_type = args.task_type

    model = DeepFM(json_path=json_path)

    if task_type == "train":
        model.train()
    if task_type == "evaluate":
        result = model.evaluate()
        print("Model metrics : ", result)
