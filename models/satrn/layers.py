import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.dense1 = nn.Linear(d_model, d_model // 2)
        self.dense2 = nn.Linear(d_model // 2, d_model * 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, _, height, width = x.shape # [B, 200, 62, 62]
        #batch_size, height, width = x.shape
        h_encoding = self.get_position_encoding(height, self.d_model)
        w_encoding = self.get_position_encoding(width, self.d_model)
        h_encoding = h_encoding.unsqueeze(1)
        w_encoding = w_encoding.unsqueeze(0)
        h_encoding = torch.tile(h_encoding.unsqueeze(0), [batch_size, 1, 1, 1])
        w_encoding = torch.tile(w_encoding.unsqueeze(0), [batch_size, 1, 1, 1])
        
        # Adaptive 2D positional encoding
        inter = torch.mean(x, axis=[2,3]) # [B, hidden] # [B, 200]
        inter = self.dense1(inter)
        inter = self.relu(inter)
        inter = self.dropout(inter)

        alpha = self.dense2(inter)
        alpha = self.sigmoid(alpha) # [B, 400]
        alpha = torch.reshape(alpha, [-1, 2, 1, self.d_model]) #[2, 2, 1, 200]
        pos_encoding = alpha[:, 0:1, :, :] * h_encoding \
                        + alpha[:, 1:2, :, :] * w_encoding
        x = x.permute(0, 2, 3, 1)
        x = x + pos_encoding

        shape = (-1, height, width, self.d_model)
        x = torch.reshape(x, (-1, height * width, self.d_model))
        x = self.dropout(x)
        return self.dropout(x), shape

    def get_position_encoding(self, length, hidden_size,
                                min_timescale=1.0, max_timescale=1.0e4):
        position = torch.arange(0, length)
        num_timescales = hidden_size // 2
        log_timescale_increment = (
            torch.log(torch.tensor(max_timescale) / torch.tensor(min_timescale)) / (num_timescales - 1))
        inv_timescales = min_timescale * \
            torch.exp(torch.arange(0, num_timescales) * -log_timescale_increment)
        #print('inv, pos: ', inv_timescales.shape, position.shape)       

        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
 
        return signal.to(device)


class ShallowCNN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, 256, 3, stride=(1, 1))
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.conv2 = nn.Conv2d(256, out_channels, 3, stride=(1, 1))
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        # self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=(1, 1))
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, 3, stride=(1, 1))
        # self.pool2 = nn.MaxPool2d(2, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=(1, 1))
        # self.pool3 = nn.MaxPool2d(2, stride=2)
        # self.conv4 = nn.Conv2d(256, 512, 3, stride=(1, 1))
        # self.pool4 = nn.MaxPool2d(2, stride=2)     
        # self.conv5 = nn.Conv2d(512, out_channels, 3, stride=(1, 1))
        # self.pool5 = nn.MaxPool2d(2, stride=2)       

        vgg16 = torchvision.models.vgg16(pretrained=True)
        modules = list(vgg16.children())[:-4]
        self.vgg16 = nn.Sequential(*modules)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)

        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.conv4(x)
        # x = self.pool4(x) 
        # x = self.conv5(x)
        # x = self.pool5(x)       

        x = self.vgg16(x) 
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


class LocalityAwareFeedForward(nn.Module):

    def __init__(self, in_channels, out_channels, dim_feedforward):
        super(LocalityAwareFeedForward, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels, dim_feedforward, 1, stride=(1, 1))
        self.depthwise_conv = DepthwiseConv(dim_feedforward, dim_feedforward, 3, stride=(1, 1))
        self.conv2 = nn.Conv2d(dim_feedforward * 3, out_channels, 1, stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x) # [B, 1536, 62, 62]
        x = self.conv2(x)
        #print('embed?', x.shape)
        return x


class DepthwiseConv(nn.Module):

    def __init__(self, nin, nout, kernels_per_layer, stride):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, stride=stride, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out    