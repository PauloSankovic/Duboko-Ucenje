import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.add_module('batch_norm_2d', nn.BatchNorm2d(num_features=num_maps_in))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv_2d', nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=(k, k), bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size

        self.brc1 = _BNReluConv(num_maps_in=input_channels, num_maps_out=emb_size, k=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.brc2 = _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.brc3 = _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def get_features(self, img):
        x = self.brc1(img)
        x = self.mp1(x)
        x = self.brc2(x)
        x = self.mp2(x)
        x = self.brc3(x)
        x = self.gap(x)
        x = x.reshape((img.size(dim=0), self.emb_size))
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        return x

    def loss(self, anchor, positive, negative, margin: float = 1.0):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        distance_positive = (a_x - p_x).norm(dim=1, p=2)
        distance_negative = (a_x - n_x).norm(dim=1, p=2)
        loss = F.relu(distance_positive - distance_negative + margin)
        return loss.mean()
