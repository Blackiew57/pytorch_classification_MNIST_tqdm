import torch
from torch import nn


class MNISTModel(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*7*7,
                      out_features=num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = nn.Linear(in_features=x.shape[1], out_features=10)(x)
        return x
    
# if __name__ == "__main__":
#     model = MNISTModel(input_shape=1, hidden_units=10, num_classes=10)
#     x = torch.randn(1, 1, 28, 28)
#     x = model(x)
#     print(x.shape)


