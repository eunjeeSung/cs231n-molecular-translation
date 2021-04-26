import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from models.resnet_lstm import DecoderRNN
from .layers import PositionalEncoding, ShallowCNN
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoder import TransformerDecoder, TransformerDecoderLayer #DecoderRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


class SATRNTrnasformer(nn.Module):
    """Base code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, ntoken, encoder_dim, nhead, dim_feedforward, nlayers,
                        seq_length, decoder_dim, stoi, batch_size, embed_size, dropout=0.5):
        super(SATRNTrnasformer, self).__init__()
        self.model_type = 'Transformer'

        self.encoder = ShallowCNN(3, encoder_dim) #nn.Embedding(ntoken, ninp)       
        self.pos_encoder1 = PositionalEncoding(encoder_dim, dropout)

        encoder_layers = TransformerEncoderLayer(encoder_dim, nhead, dim_feedforward, dropout)        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, stoi)

        self.embed = nn.Embedding(num_embeddings=embed_size, embedding_dim=decoder_dim)
        self.pos_encoder2 = PositionalEncoding(decoder_dim, dropout)
        #decoder_layer = TransformerDecoderLayer(decoder_dim, nhead)
        #self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers, decoder_dim, stoi)
        
        decoder_layer = nn.TransformerDecoderLayer(decoder_dim, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers//2)
    
        self.linear = nn.Linear(decoder_dim, ntoken)
        self.out = nn.Softmax()

        self.ntoken = ntoken
        self.stoi = stoi

        # self.decoder = DecoderRNN(
        #     embed_size=encoder_dim,
        #     vocab_size = ntoken,
        #     attention_dim=300,
        #     encoder_dim=encoder_dim,
        #     decoder_dim=decoder_dim
        # )

        self.seq_length = seq_length
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.batch_size = batch_size
        #self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     self.transformer_encoder.weight.data.uniform_(-initrange, initrange)
    #     self.transformer_decoder.bias.data.zero_()
    #     self.transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, captions):
        src = self.encoder(src) # [B, 200, 62, 62]      
        src, shape = self.pos_encoder1(src)
        memory = self.transformer_encoder(src, shape)          
        
        init_ids = torch.ones([self.batch_size, 1], dtype=torch.int32) * self.stoi['<sos>']
        captions = torch.cat([init_ids.to(device), captions], axis=1)[:, :-1]

        embed = self.embed(captions)       
        pos = self.pos_encoder2.get_position_encoding(embed.shape[1], self.decoder_dim)        
        #pos = self.pos_encoder2(embed)
        
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(device)
        tgt = embed + pos
        tgt = tgt.permute(1, 0, 2)
        
        output = self.transformer_decoder(tgt, memory, tgt_mask)   
        output = self.linear(output)
        #output = self.out(output)      
        output = output.permute(1, 0, 2)

        return output

    def evaluate(self, src):
        #src = torch.normal(0, 1, size=src.shape).to(device)
        src = self.encoder(src)
        #print('src after encoder:', src[0, 0, 20:31, 20:31]) # [1, 256, 62, 62]
        #print(src)
        #src = torch.normal(0, 4, size=src.shape).to(device)
        src, shape = self.pos_encoder1(src)
        #print('src after pos_encoder1:', src[0, 1800:1922, 0]) # [1, 3844, 256]
        
        #src = torch.normal(0, 2, size=src.shape).to(device)
        memory = self.transformer_encoder(src, shape)        
        #print('memory:', memory[0, 1800:1922, 0]) # [1, 3844, 256]

        inputs, logits = [self.stoi['<sos>']], []
        for i in range(self.seq_length):
            tgt = torch.LongTensor([inputs]).view(-1,1).to(device)       
            tgt_mask = self.generate_square_subsequent_mask(i+1).to(device)       

            tgt = self.embed(tgt)
            pos = self.pos_encoder2.get_position_encoding(tgt.shape[1], self.decoder_dim)
            
            output = self.transformer_decoder(
                tgt=tgt, 
                memory=memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=None)
            
            output = self.linear(output)
            output = self.out(output)
            output = output[-1] # the last timestep
            values, indices = output.max(dim=-1)
            pred_token = indices.item()
            inputs.append(pred_token)   
            logits.append(output)

        pred_indices = torch.tensor([inputs])
        #logits = torch.tensor(logits)

        #return  logits, inputs[1:]
        return pred_indices