import torch
import torch.nn as nn
import torch.nn.functional as F

class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()

        self.inLinear       = 30
        self.outLinear      = 128
        self.inChannels     = [128, 256, 512]
        self.outChannels    = [256, 512, 71]
        self.dropout        = 0.6

        self.Linear1 = nn.Sequential(
            nn.Linear(self.inLinear, self.outLinear), 
            nn.BatchNorm1d(self.outLinear),
            nn.LeakyReLU(), 
            nn.Dropout(p=self.dropout)
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(self.inChannels[0], self.outChannels[0]), 
            nn.BatchNorm1d(self.outChannels[0]),
            nn.LeakyReLU(), 
            nn.Dropout(p=self.dropout)
        )
        
        self.linear3 = nn.Sequential(
            nn.Linear(self.inChannels[1], self.outChannels[1]),
            nn.BatchNorm1d(self.outChannels[1]),
            nn.LeakyReLU(), 
            nn.Dropout(p=self.dropout)
        )
        
        self.linear4 = nn.Sequential(
            nn.Linear(self.inChannels[2], self.outChannels[2]),
            nn.BatchNorm1d(self.outChannels[2]),
            nn.Tanh(), 
        )

    def forward(self, x):
        x = self.Linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = x.unsqueeze(1)
        return x

def generatorNet():
    return NetG()