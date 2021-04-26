import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .layers import ConvPoolBlock, DoubleConv, Down, Up, OutConv, double_conv, vgg16_bn


# class EncoderCNNtrain18(nn.Module):
#     def __init__(self, in_channels, encoder_dim):
#         super(EncoderCNNtrain18, self).__init__()
#         resnet = torchvision.models.resnet18()
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)        

#     def forward(self, images):
#         features = self.resnet(images) 
#         features = features.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
#         features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)
#         return features


# class EncoderCNNtrain18(nn.Module):
#     def __init__(self, in_channels, encoder_dim):
#         super(EncoderCNNtrain18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, encoder_dim, 5, stride=(1, 1))
#         self.conv2 = nn.Conv2d(encoder_dim, encoder_dim, 5, stride=(1, 1))
#         self.conv_pool1 = ConvPoolBlock(encoder_dim, encoder_dim)
#         self.conv_pool2 = ConvPoolBlock(encoder_dim, encoder_dim)
#         self.conv_pool3 = ConvPoolBlock(encoder_dim, encoder_dim)
#         self.conv_pool4 = ConvPoolBlock(encoder_dim, encoder_dim)

#     def forward(self, images):
#         features = self.conv1(images)
#         features = self.conv2(features)
#         features = self.conv_pool1(features)   
#         features = self.conv_pool2(features)
#         features = self.conv_pool3(features)
#         features = self.conv_pool4(features)                     #(batch_size,512,8,8)

#         features = features.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
#         features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)
           
#         return features        


# class EncoderCNNtrain18(nn.Module):
#     def __init__(self, in_channels, encoder_dim, bilinear=True):
#         super(EncoderCNNtrain18, self).__init__()
#         self.in_channels = in_channels
#         self.encoder_dim = encoder_dim
#         self.bilinear = bilinear

#         self.inc = DoubleConv(in_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, encoder_dim)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         features = self.outc(x)

#         features = features.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
#         features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)
         
#         return features


class EncoderCNNtrain18(nn.Module):
    def __init__(self, in_channels, encoder_dim, pretrained=True, freeze=True):
        super(EncoderCNNtrain18, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, encoder_dim)
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        features = self.upconv4(y)

        features = features.permute(0, 2, 3, 1)                           #(batch_size,8,8,512)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,64,512)

        return features