from torch import nn 

class FireDetectorMain3D(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, padding): 
        super().__init__()
        T, H, W = kernel_size

        self.spatial_conv = nn.Conv3d(
            in_channels,  out_channels , kernel_size=(1 , H, W), padding=(0, padding , padding),
        )

        self.relu = nn.ReLU()
        
        self.temporal_conv = nn.Conv3d(
            out_channels , out_channels , kernel_size=(T , 1, 1) , padding=(padding, 0 , 0)
        )
    
    def forward(self, x): 
        x = self.spatial_conv(x)
        x = self.relu(x) 
        x = self.temporal_conv(x) 
        return x 

class Residual(nn.Module): 
    def __init__(self, channels , kernel_size, padding =1):
        super().__init__()
        self.conv1 = FireDetectorMain3D(channels, channels, kernel_size , padding)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = FireDetectorMain3D(channels ,channels, kernel_size , padding)
        self.norm2 = nn.LayerNorm(channels)
        self.relu = nn.ReLU()

    def forward(self, x) : 
        Residual = x 
        out = self.conv1(x) #(N, C, D, H, W)
        out = out.permute(0, 2 , 3 , 4 , 1)
        out = self.norm1(out)
        out= out.permute(0 , 4 , 1 , 2, 3)
        # out = self.relu(out)

        out = self.conv2(out)
        out = out.permute(0, 2 , 3 , 4 , 1)
        out = self.norm2(out)
        out= out.permute(0 , 4 , 1 , 2, 3)

        out += Residual
        out = self.relu(out)
        return out 
    
class FireDetectorWithResidual(nn.Module): 
        def __init__(self , in_channels =3 , num_clases =1) :
            super().__init__()
            self.initial_conv = FireDetectorMain3D(in_channels, 16 , kernel_size=(3, 7, 7) , padding=1)
            self.bn = nn.BatchNorm3d(16)
            self.relu = nn.ReLU() 
            self.pool1 = nn.MaxPool3d((1 , 2 ,2))
            self.res_block1 = Residual(16, kernel_size=(3 ,3 ,3))
            self.pool2 = nn.MaxPool3d((2 ,2 ,2))

            self.res_block2 = Residual(16, kernel_size=(3 ,3 ,3))
            self.adaptive_pool = nn.AdaptiveAvgPool3d((1 ,1,1))

            self.classifier = nn.Linear(16 , num_clases)

        def forward(self  , x): 
            x = self.initial_conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool1(x)

            x = self.res_block1(x)
            x = self.pool2(x)

            x = self.res_block2(x)
            x = self.adaptive_pool(x)

            x = x.flatten(1)
            x = self.classifier(x)

            return x