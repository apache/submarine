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
import torch

import submarine


class LinearNNModel(torch.nn.Module):
    def __init__(self):
        super(LinearNNModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == "__main__":
    net = LinearNNModel()
    submarine.save_model(
        model_type="pytorch",
        model=net,
        artifact_path="pytorch-nn-model",
        registered_model_name="simple-nn-model",
        input_dim=[2],
        output_dim=[1],
    )
