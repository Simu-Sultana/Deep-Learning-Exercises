import torch.nn as nn

def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.l1 = ResBlock(64,  64, 1)
        self.l2 = ResBlock(64, 128, 2)
        self.l3 = ResBlock(128, 256, 2)
        self.l4 = ResBlock(256, 512, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return self.sig(x)
