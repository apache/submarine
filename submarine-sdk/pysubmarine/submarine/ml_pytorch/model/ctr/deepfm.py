from .field_linear import FieldLinear
from .field_embedding import FieldEmbedding 
from .pairwise_interaction import PairwiseInteraction 

import torch 
from torch import nn 

class DeepFM(nn.Module):   
    def __init__(self, field_dims, embedding_dim, out_features, hidden_units, dropout_rates): 
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
        emb = self.field_embedding(x) # (batch_size, num_fields, embedding_dim) 
        return self.field_linear(x) + self.pairwise_interaction(emb) + self.dnn(torch.flatten(emb, start_dim=1))




class DNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_units: list, dropout_rates: list):
        super().__init__()
        *layers, out_layer = list(zip([in_features, *hidden_units], [*hidden_units, out_features]))
        self.net = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(in_features=i, out_features=o), 
                nn.BatchNorm1d(num_features=o), 
                nn.ReLU(), 
                nn.Dropout(p=p)
            ) for (i, o), p in zip(layers, dropout_rates)), 
            nn.Linear(*out_layer) 
        )
    def forward(self, x): 
        """
        :param x: torch.FloatTensor (batch_size, in_features) 
        """
        return self.net(x) 
