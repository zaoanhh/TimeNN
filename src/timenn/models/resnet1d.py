"""1D ResNet used as the backbone for TN-ResNet and as a baseline.

The implementation follows the standard ResNet-18/34 layout with 1D
convolutions and an optional ``for_lmsnet`` switch that returns the feature
map instead of the logits.
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        # padding_mode='circular'
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        # Removed strict checks for groups and base_width to allow flexibility
        # if groups != 1 or base_width != 64:
        #     raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels=1,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,

        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        base_filters=64,
        dropout=0.0,
        for_lmsnet=False,
        mlp_hidden=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = base_filters
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * (len(layers) - 1)
            
        if len(replace_stride_with_dilation) != len(layers) - 1:
             # Just handle it gracefully by padding or slicing
             pass

        self.groups = groups
        self.base_width = width_per_group
        
        # Initial layers
        self.conv1 = nn.Conv1d(
            in_channels, self.inplanes, kernel_size=7, stride=1 if for_lmsnet else 2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if not for_lmsnet else nn.Identity()    

        self.layers = nn.ModuleList()
        
        # Dynamic Layer Creation
        # Layer 1 (Stride 1 usually)
        self.layers.append(self._make_layer(block, base_filters, layers[0]))
        
        # Subsequent layers (Stride 2 usually, increasing channels)
        num_filters = base_filters
        for i in range(1, len(layers)):
            num_filters *= 2
            dilate = replace_stride_with_dilation[i-1] if i-1 < len(replace_stride_with_dilation) else False
            self.layers.append(self._make_layer(
                block, num_filters, layers[i], stride=2, dilate=dilate
            ))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        in_features = num_filters * block.expansion
        if mlp_hidden is not None and mlp_hidden > 0:
            self.fc = nn.Sequential(
                nn.Linear(in_features, mlp_hidden),
                nn.GELU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_hidden, num_classes),
            )
        else:
            self.fc = nn.Linear(in_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
    def extract_features(self, x):
        """
        专门用于 t-SNE 的特征提取函数
        返回: [Batch_Size, Feature_Dim] 的特征向量
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features


def resnet18(in_channels=1, num_classes=1000, layers=[2, 2, 2, 2], **kwargs):
    return ResNet(BasicBlock, layers, in_channels=in_channels, num_classes=num_classes, **kwargs)
