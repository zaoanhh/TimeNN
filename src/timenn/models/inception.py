"""A compact InceptionTime-style 1D classifier (baseline)."""

import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kss=[10, 20, 40], bottleneck_channels=32, activation='relu'):
        super().__init__()
        self.kss = kss
        self.bottleneck_channels = bottleneck_channels
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

        if in_channels > 1 and bottleneck_channels > 0:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            current_in_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            current_in_channels = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(current_in_channels, out_channels, kernel_size=k, padding=k // 2, bias=False) for k in kss
        ])
        
        # MaxPool branch
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        
        self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x_in = x
        
        outs = []
        for conv in self.convs:
            out = conv(x_in)
            # k // 2 padding preserves the input length for odd kernels.
            if out.shape[-1] != x_in.shape[-1]:
                 out = out[..., :x_in.shape[-1]]
            outs.append(out)
            
        # MaxPool branch: pool the input, then apply a 1x1 convolution.
        mp_out = self.maxpool_conv(input_tensor)
        if mp_out.shape[-1] != x_in.shape[-1]:
             mp_out = mp_out[..., :x_in.shape[-1]]
             
        outs.append(mp_out)
        
        out = torch.cat(outs, dim=1)
        out = self.bn(out)
        out = self.activation(out)
        return out

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_modules=3, kss=[10, 20, 40], bottleneck_channels=32, activation='relu', use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.modules_list = nn.ModuleList()
        
        for i in range(num_modules):
            self.modules_list.append(
                InceptionModule(
                    in_channels if i == 0 else out_channels * 4, 
                    out_channels, 
                    kss=kss, 
                    bottleneck_channels=bottleneck_channels, 
                    activation=activation
                )
            )
            
        if use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, 1, bias=False) if in_channels != out_channels * 4 else nn.Identity(),
                nn.BatchNorm1d(out_channels * 4)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        for m in self.modules_list:
            x = m(x)
            
        if self.use_residual:
            res = self.shortcut(res)
            x = x + res
            x = self.relu(x)
        return x

class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks=2, num_modules_per_block=3, out_channels=32, kss=[10, 20, 40], bottleneck_channels=32, use_residual=True):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(num_blocks):
            self.blocks.append(
                InceptionBlock(
                    current_channels, 
                    out_channels, 
                    num_modules=num_modules_per_block, 
                    kss=kss, 
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual
                )
            )
            current_channels = out_channels * 4
            
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        # x: [B, C, T]
        for block in self.blocks:
            x = block(x)
            
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
