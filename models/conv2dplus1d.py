from torch import nn 

class FireDetectorConv2D1D(nn.Module): 
    def __init__(self, in_channels = 3 , num_classes=1): 
        super().__init__() 
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size = 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2),

            nn.Conv2d(32 , 64, kernel_size = 3 , padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.temporal_extractor = nn.Sequential(
            nn.Conv1d(64, 128 , 3 ,padding = 1 ),
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool1d(1) 
        )

        self.clasifier = nn.Linear(128, num_classes)

    def forward(self , x): 
            B , C , T , H , W = x.size()
            x = x.permute(0, 2, 1, 3 ,4)  # (B, T, C, H, W)
            x = x.reshape(B * T , C , H , W)
            x = self.spatial_extractor(x) # (B*T, 64, 1, 1)
            x = x.view(B , T , 64)  # (B, T, 64)

            x = x.permute(0 , 2 , 1) # (B , 64, T)
            x = self.temporal_extractor(x) # (B , 128, 1)
            x = x.squeeze(-1) # (B , 128)

            return self.clasifier(x)