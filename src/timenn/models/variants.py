"""Baseline model zoo for the motivation experiment.

Contains a plain CNN, a bidirectional LSTM/GRU and a factory for the two
anti-aliased ResNet depths (ResNet-18/34).
"""


import torch
import torch.nn as nn
from timenn.models.resnet1d_kaiming import Resnet

class SimpleCNN1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_channels, num_classes, rnn_type='LSTM', hidden_size=128, num_layers=2, dropout=0.5):
        super(SimpleRNN, self).__init__()
        self.rnn_type = rnn_type
        
        # RNN expects [Batch, Seq, Features]; input is [Batch, Channel, Time].
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        else:
            raise ValueError("rnn_type must be LSTM or GRU")
            
        self.fc = nn.Linear(hidden_size * 2, num_classes) # *2 for bidirectional

    def forward(self, x):
        # x: [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C]
        
        if self.rnn_type == 'LSTM':
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
            
        # Global average pooling over time.
        # out: [B, T, hidden*2]
        out = torch.mean(out, dim=1) # [B, hidden*2]
        out = self.fc(out)
        return out

def get_resnet_model(resnet_type, input_channels, num_classes):
    """
    Factory for ResNet1D variants.
    """
    if resnet_type == 'resnet18':
        return Resnet(
            n_channels=input_channels,
            outsize=num_classes,
            n_filters_list=[64, 128, 256, 512],
            kernel_size_list=7,
            n_resblocks_list=[2, 2, 2, 2],
            resblock_kernel_size_list=3,
            downfactor_list=[2, 2, 2, 2],
            downorder_list=[0, 0, 0, 0],
            drop1=0.0, drop2=0.0,
            fc_size=512
        )
    elif resnet_type == 'resnet34':
        return Resnet(
            n_channels=input_channels,
            outsize=num_classes,
            n_filters_list=[64, 128, 256, 512],
            kernel_size_list=7, 
            n_resblocks_list=[3, 4, 6, 3],
            resblock_kernel_size_list=3,
            downfactor_list=[2, 2, 2, 2],
            downorder_list=[0, 0, 0, 0], 
            drop1=0.0, drop2=0.0,
            fc_size=512
        )
    else:
        raise ValueError(f"Unknown resnet type: {resnet_type}")
