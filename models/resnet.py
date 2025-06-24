import torch 
from torch import nn
import torchvision.models as models


class ResNet18(nn.Module):
    """
    For CIFAR10
    """
    def __init__(self, num_channel=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)

        self.resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x
    
class ResNet34(nn.Module): 
    """
    For CIFAR100
    """
    def __init__(self, num_channel=3, num_classes=100):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(weights=None)

        self.resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x
    