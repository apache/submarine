from submarine.ml.pytorch.layers.core import FieldLinear
from submarine.ml.pytorch.layers.core import FieldEmbedding
from submarine.ml.pytorch.layers.core import PairwiseInteraction
from submarine.ml.pytorch.layers.core import DNN

import torch
from torch import nn
from submarine.ml.pytorch.model.base_pytorch_model import BasePyTorchModel


class DeepFM(BasePyTorchModel):
    def model_fn(self, params):
        super().model_fn(params)
        return _DeepFM(**self.params['model']['kwargs'])


class _DeepFM(nn.Module):
    def __init__(self, field_dims, embedding_dim, out_features,
                 hidden_units, dropout_rates, **kwargs):
        super().__init__()
        self.field_linear = FieldLinear(field_dims=field_dims, out_features=out_features)
        self.field_embedding = FieldEmbedding(field_dims=field_dims, embedding_dim=embedding_dim)
        self.pairwise_interaction = PairwiseInteraction()
        self.dnn = DNN(
            in_features=len(field_dims)*embedding_dim,
            out_features=out_features,
            hidden_units=hidden_units,
            dropout_rates=dropout_rates
        )

    def forward(self, x):
        """
        :param x: torch.LongTensor (batch_size, num_fields)
        """
        emb = self.field_embedding(x)  # (batch_size, num_fields, embedding_dim)
        linear_logit = self.field_linear(x)
        fm_logit = self.pairwise_interaction(emb)
        deep_logit = self.dnn(torch.flatten(emb, start_dim=1))

        return linear_logit + fm_logit + deep_logit
