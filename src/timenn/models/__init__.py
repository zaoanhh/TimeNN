"""Network definitions.

``lms_net`` contains the core :class:`~timenn.models.lms_net.LMSBlock` (the
Time-NN block).  The remaining modules wrap it into complete classifiers or
provide the baselines used in the paper.
"""

from timenn.models.lms_net import FilterConfig, LMSBlock, LMSNet
from timenn.models.lms_cnn import LMSCNN, LMSCNNConfig
from timenn.models.lms_resnet import LMSResNet, LMSResNetConfig
from timenn.models.lms_tenc1 import LMSTEnc1Transformer
from timenn.models.lms_tproj import LMSTProjTransformer
from timenn.models.simple_cnn import SimpleCNNClassifier, SimpleCNN1D, SimpleCNNConfig
from timenn.models.resnet1d import ResNet, BasicBlock, resnet18
from timenn.models.transformer import TimeSeriesTransformer, transformer_s
from timenn.models.inception import InceptionTime

__all__ = [
    "FilterConfig",
    "LMSBlock",
    "LMSNet",
    "LMSCNN",
    "LMSCNNConfig",
    "LMSResNet",
    "LMSResNetConfig",
    "LMSTEnc1Transformer",
    "LMSTProjTransformer",
    "SimpleCNNClassifier",
    "SimpleCNN1D",
    "SimpleCNNConfig",
    "ResNet",
    "BasicBlock",
    "resnet18",
    "TimeSeriesTransformer",
    "transformer_s",
    "InceptionTime",
]
