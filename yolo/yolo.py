import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLO(nn.Module):
    def __init__(self, args):
        super(YOLO, self).__init__()

        self.args = args
        
        # num of grid cell
        self.S = self.args.S
        # num of bounding box per cell
        self.B = self.args.B
        # num of classes
        self.C = self.args.C

        self.feature_cfg = [
            [3, 64, 7, 2, 3],
            'MAXPOOL',
            [64, 192, 3, 1, 1],
            'MAXPOOL',
            [192, 128, 1, 1, 0],
            [128, 256, 3, 1, 1],
            [256, 256, 1, 1, 0],
            [256, 512, 3, 1, 1],
            'MAXPOOL',
            [512, 512, 1, 1, 0],
            [512, 1024, 3, 1, 1],
            [1024, 1024, 3, 1, 1],
            [1024, 1024, 3, 1, 1],
            [1024, 1024, 3, 2, 1],
            'MAXPOOL',
            [1024, 1024, 3, 1, 1],
            [1024, 1024, 3, 1, 1]
        ]
        self.classifier_cfg = [
            [1024, 4096, 7, 1, 0],
            'DROPOUT',
            [4096, 4096, 1, 1, 0],
            'DROPOUT',
            [4096, (self.B * 5 + self.C) * self.S * self.S, 1, 1, 0]
        ]
        self.features = stack_layers(self.feature_cfg)
        self.classifier = stack_layers(self.classifier_cfg)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, 7, 7, self.B * 5 + self.C)
        return x

def stack_layers(cfg):
    layers = []
    
    for layer in cfg:
        if layer == 'MAXPOOL':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer == 'DROPOUT':
            layers.append(nn.Dropout())
        else:
            in_channel, out_channel, kernel_size, stride, padding = layer
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,\
                                                        stride=stride, padding=padding))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
    
    return nn.Sequential(*layers)
