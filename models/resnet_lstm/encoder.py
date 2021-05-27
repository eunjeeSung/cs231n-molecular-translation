import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .layers import ConvPoolBlock, DoubleConv, Down, Up, OutConv, \
                    double_conv, vgg16_bn, LocalityAwareFeedForward, ConvBatchReLUBlock, MBConv6, \
                    SpatialAttention, ChannelAttention, Hourglass, HRNet

# Setting 30: FaceNet + (1,2 strided) + FPN reconstruction
class EncoderCNNtrain18(nn.Module):
    def __init__(self, in_channels, encoder_dim):
        super(EncoderCNNtrain18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=(1, 2))
        self.pool1 = nn.MaxPool2d(3, 2)
        self.ln1 = nn.LayerNorm([76, 78])

        self.sa = SpatialAttention()

        self.conv2a = nn.Conv2d(64, 64, 1, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
        self.ln2 = nn.LayerNorm([76, 78])
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3a = nn.Conv2d(192, 192, 1, stride=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2)

        self.conv4a = nn.Conv2d(384, 384, 1, stride=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5a = nn.Conv2d(384, 256, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv6a = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, encoder_dim, 3, padding=1)
        self.project = nn.Conv2d(384, encoder_dim, 1)
        self.pool4 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.ln1(x)
        x = self.sa(x) * x

        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        res = x
        x = self.conv4a(x)
        x = self.conv4(x)
        x = self.conv5a(x)
        x = self.conv5(x)
        x = self.conv6a(x)
        x = self.conv6(x)
        x += self.project(res)
        x = self.pool4(x)

        features = x.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)
        return features