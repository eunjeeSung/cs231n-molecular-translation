import torch
import torch.nn as nn
import torchvision

from .encoder import EncoderCNNtrain18
from .decoder import DecoderRNN
from .layers import Classifier

class EncoderDecodertrain18(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNNtrain18(3, encoder_dim)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        self.m_classifier = Classifier(64 * 512, 5)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def evaluate(self, images, stoi, itos):
        features = self.encoder(images)
        captions = self.decoder.generate_caption(features, stoi, itos)
        return captions
