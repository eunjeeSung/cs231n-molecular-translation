import torch
import torch.nn as nn
import torchvision


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
        return features