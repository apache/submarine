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


# pylint: disable=W0223
class FeatureLinear(nn.Module):

    def __init__(self, num_features, out_features):
        """
        :param num_features: number of total features.
        :param out_features: The number of output features.
        """
        super().__init__()
        self.weight = nn.Embedding(num_embeddings=num_features,
                                   embedding_dim=out_features)
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, feature_idx, feature_value):
        """
        :param feature_idx: torch.LongTensor (batch_size, num_fields)
        :param feature_value: torch.LongTensor (batch_size, num_fields)
        """
        return torch.sum(
            self.weight(feature_idx) * feature_value.unsqueeze(dim=-1),
            dim=1) + self.bias


class FeatureEmbedding(nn.Module):

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.weight = nn.Embedding(num_embeddings=num_features,
                                   embedding_dim=embedding_dim)

    def forward(self, feature_idx, feature_value):
        """
        :param feature_idx: torch.LongTensor (batch_size, num_fields)
        :param feature_value: torch.LongTensor (batch_size, num_fields)
        """
        return self.weight(feature_idx) * feature_value.unsqueeze(dim=-1)


class PairwiseInteraction(nn.Module):

    def forward(self, x):
        """
        :param x: torch.Tensor (batch_size, num_fields, embedding_dim)
        """
        square_of_sum = torch.square(torch.sum(
            x, dim=1))  # (batch_size, embedding_dim)
        # (batch_size, embedding_dim)
        sum_of_square = torch.sum(torch.square(x), dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square,
                               dim=1,
                               keepdim=True)  # (batch_size, 1)


class DNN(nn.Module):

    def __init__(self, in_features, out_features, hidden_units, dropout_rates):
        super().__init__()
        *layers, out_layer = list(
            zip([in_features, *hidden_units], [*hidden_units, out_features]))
        self.net = nn.Sequential(
            *(nn.Sequential(nn.Linear(in_features=i, out_features=o),
                            nn.BatchNorm1d(num_features=o), nn.ReLU(),
                            nn.Dropout(p=p))
              for (i, o), p in zip(layers, dropout_rates)),
            nn.Linear(*out_layer))

    def forward(self, x):
        """
        :param x: torch.FloatTensor (batch_size, in_features)
        """
        return self.net(x)
