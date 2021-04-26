import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LocalityAwareFeedForward
from util import _get_clones

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class TransformerEncoder(nn.Module):
    """Code copied from the PyTorch official implementation.
    TransformerEncoder is a stack of N encoder layers
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, stoi, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.stoi = stoi

    def forward(self, src, shape, src_pad_key=None, mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layers in turn.
        """
        output = src

        src_pad_key = self.stoi['<pad>']
        input_msk = (src != src_pad_key).unsqueeze(1)

        for mod in self.layers:
           output = mod(output, shape, src_mask=input_msk, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Base code from the PyTorch official implementation.
    """

    def __init__(self, embed_dim, nhead, dim_feedforward=2048, dropout=0.1,
                        activation="relu", layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)             
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.locality_ffn = LocalityAwareFeedForward(embed_dim, embed_dim, dim_feedforward)

        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        self.embed_dim = embed_dim
        #self.dropout2 = nn.Dropout(dropout)

        #self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, shape, src_mask, src_key_padding_mask):
        src = self.norm1(src)
        src = src.permute(1, 0, 2)

        src2 = self.self_attn(src, src, src)[0]
        src2 = self.dropout1(src2)
        src = src + src2
        src = src.permute(1, 0, 2)

        src2 = self.norm2(src)
        src2 = torch.reshape(src, shape)
        src2 = src2.permute(0, 3, 1, 2)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.locality_ffn(src2)
        #src2 = src2.permute(0, 2, 3, 1)      
        src2 = torch.reshape(src2, (-1, shape[1]*shape[2], self.embed_dim))      
        #print('src/src2:', src.shape, src2.shape)        
        src = src + src2

        #src = src + self.dropout2(src2)
        src = self.norm3(src)
        return src
