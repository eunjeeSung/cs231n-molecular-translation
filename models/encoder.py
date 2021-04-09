import torch
import torch.nn as nn
import torchvision

from models.decoder import DecoderRNN


class EncoderCNNtrain18(nn.Module):
    def __init__(self):
        super(EncoderCNNtrain18, self).__init__()
        resnet = torchvision.models.resnet18()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,512,8,8)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)
        #print(features.shape)
        return features
    

class EncoderDecodertrain18(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNNtrain18()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs