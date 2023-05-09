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

        self.features = self.stack_conv()
        self.classifier = self.stack_linear()
        self.sigmoid = nn.Sigmoid()

        # for layer in self.features.module():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.normal_(layer.weight, mean=0, std=0.1)
        
        # for layer in self.classifier.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.normal_(layer.weight, mean=0, std=0.1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        x = x.view(-1, 7, 7, self.B * 5 + self.C)
        
        x[:, :, :, 4] = self.sigmoid(x[:, :, :, 4])
        x[:, :, :, 10:] = self.sigmoid(x[:, :, :, 10:])

        return x

    def stack_conv(self):
        backbone = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Flatten()
        )

        return backbone
    
    def stack_linear(self):
        classifier = nn.Sequential(
            nn.Linear(self.S * self.S * 1024, 4096),
            nn.Dropout(self.args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

        return classifier

