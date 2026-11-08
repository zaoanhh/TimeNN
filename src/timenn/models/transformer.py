"""A small Transformer encoder for time-series classification (baseline)."""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # [1, d_model, max_len] to match [B, C, T] convention usually
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, d_model, T]
        x = x + self.pe[:, :, :x.size(2)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_classes, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dim_feedforward=256, 
                 dropout=0.1, 
                 max_len=5000,
                 patch_size=10,
                 stride=10):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection: [B, C, T] -> [B, d_model, T]
        self.input_projection = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=stride)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [B, C, T]
        
        # Project to d_model
        x = self.input_projection(x) # [B, d_model, T]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects [B, T, d_model] for batch_first=True
        x = x.permute(0, 2, 1) # [B, T, d_model]
        
        x = self.transformer_encoder(x)
        
        # Back to [B, d_model, T] for pooling
        x = x.permute(0, 2, 1)
        
        x = self.avgpool(x).squeeze(2) # [B, d_model]
        
        x = self.fc(x)
        return x
    
    def extract_features(self, x):
        """
        Returns: [Batch_Size, Feature_Dim]
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        features = self.avgpool(x).squeeze(2)
        return features

def transformer_s(in_channels=1, num_classes=1000, **kwargs):
    # Small default configuration
    return TimeSeriesTransformer(in_channels=in_channels, num_classes=num_classes, **kwargs)
