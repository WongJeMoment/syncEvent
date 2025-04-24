import torch.nn as nn

class EdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
