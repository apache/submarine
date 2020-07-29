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

import torch
from torch import nn

from submarine.ml.pytorch.layers.core import (DNN, FeatureEmbedding,
                                              FeatureLinear,
                                              PairwiseInteraction)
from submarine.ml.pytorch.model.base_pytorch_model import BasePyTorchModel


class DeepFM(BasePyTorchModel):

    def model_fn(self, params):
        super().model_fn(params)
        return _DeepFM(**self.params['model']['kwargs'])


# pylint: disable=W0223
class _DeepFM(nn.Module):

    def __init__(self, num_fields, num_features, embedding_dim, out_features,
                 hidden_units, dropout_rates, **kwargs):
        super().__init__()
        self.feature_linear = FeatureLinear(num_features=num_features,
                                            out_features=out_features)
        self.feature_embedding = FeatureEmbedding(num_features=num_features,
                                                  embedding_dim=embedding_dim)
        self.pairwise_interaction = PairwiseInteraction()
        self.dnn = DNN(in_features=num_fields * embedding_dim,
                       out_features=out_features,
                       hidden_units=hidden_units,
                       dropout_rates=dropout_rates)

    def forward(self, feature_idx, feature_value):
        """
        :param feature_idx: torch.LongTensor (batch_size, num_fields)
        :param feature_value: torch.LongTensor (batch_size, num_fields)
        """
        emb = self.feature_embedding(
            feature_idx,
            feature_value)  # (batch_size, num_fields, embedding_dim)
        linear_logit = self.feature_linear(feature_idx, feature_value)
        fm_logit = self.pairwise_interaction(emb)
        deep_logit = self.dnn(torch.flatten(emb, start_dim=1))

        return linear_logit + fm_logit + deep_logit
