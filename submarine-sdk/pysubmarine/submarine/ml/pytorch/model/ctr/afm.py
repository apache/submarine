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

from submarine.ml.pytorch.layers.core import (FeatureEmbedding, FeatureLinear)
from submarine.ml.pytorch.model.base_pytorch_model import BasePyTorchModel


class AFM(BasePyTorchModel):

    def model_fn(self, params):
        super().model_fn(params)
        return _AFM(**self.params['model']['kwargs'])


# pylint: disable=W0223
class _AFM(nn.Module):

    def __init__(self, num_features: int, embedding_dim: int,
                 attention_dim: int, out_features: int, dropout_rate: float,
                 **kwargs):
        super().__init__()
        self.feature_linear = FeatureLinear(num_features=num_features,
                                            out_features=out_features)
        self.feature_embedding = FeatureEmbedding(num_features=num_features,
                                                  embedding_dim=embedding_dim)
        self.attentional_interaction = AttentionalInteratction(
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            out_features=out_features,
            dropout_rate=dropout_rate)

    def forward(self, feature_idx: torch.LongTensor,
                feature_value: torch.LongTensor):
        """
        :param feature_idx: torch.LongTensor (batch_size, num_fields)
        :param feature_value: torch.LongTensor (batch_size, num_fields)
        """
        return self.feature_linear(
            feature_idx, feature_value) + self.attentional_interaction(
                self.feature_embedding(feature_idx, feature_value))


class AttentionalInteratction(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int,
                 out_features: int, dropout_rate: float):
        super().__init__()
        self.attention_score = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=attention_dim),
            nn.ReLU(), nn.Linear(in_features=attention_dim, out_features=1),
            nn.Softmax(dim=1))
        self.pairwise_product = PairwiseProduct()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=embedding_dim,
                            out_features=out_features)

    def forward(self, x: torch.FloatTensor):
        """
        :param x: torch.FloatTensor (batch_size, num_fields, embedding_dim)
        """
        x = self.pairwise_product(x)
        score = self.attention_score(x)
        attentioned = torch.sum(score * x, dim=1)
        return self.fc(self.dropout(attentioned))


class PairwiseProduct(nn.Module):

    def forward(self, x: torch.FloatTensor):
        """
        :param x: torch.FloatTensor (batch_sie, num_fields, embedding_dim)
        """
        _, num_fields, _ = x.size()

        all_pairs_product = x.unsqueeze(dim=1) * x.unsqueeze(dim=2)
        idx_row, idx_col = torch.unbind(torch.triu_indices(num_fields,
                                                           num_fields,
                                                           offset=1),
                                        dim=0)
        return all_pairs_product[:, idx_row, idx_col]
