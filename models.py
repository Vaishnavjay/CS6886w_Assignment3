import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2

class MobileNetV2CIFAR10(nn.Module):
    def __init__(self, width_mult: float = 1.0, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        mnet = MobileNetV2(width_mult=width_mult, num_classes=num_classes)

        # stride-1 stem for CIFAR-10
        first_out = int(32 * width_mult)
        mnet.features[0] = nn.Sequential(
            nn.Conv2d(3, first_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_out),
            nn.ReLU6(inplace=True),
        )

        last_channel = mnet.last_channel
        mnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(last_channel, num_classes),
        )

        self.mnet = mnet

    def forward(self, x):
        return self.mnet(x)