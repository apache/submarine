import torch 
from torch import nn 

from itertools import accumulate 


class FieldLinear(nn.Module): 
    def __init__(self, field_dims, out_features): 
        super().__init__() 
        """
        :param field_dims: List of dimensions of each field. 
        :param out_features: The number of output features. 
        """
        self.weight = nn.Embedding(num_embeddings=sum(field_dims), embedding_dim=out_features) 
        self.bias = nn.Parameter(torch.zeros((out_features,))) 
        self.register_buffer('offset', torch.as_tensor([0, *accumulate(field_dims)][:-1], dtype=torch.long))

    def forward(self, x): 
        """
        :param x: torch.LongTensor (batch_size, num_fields) 
        """
        return torch.sum(self.weight(x + self.offset), dim=1) + self.bias 


