import torch
import torch.nn as nn
import torch.nn.functional as F

def init_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
        
def _init_weights_kaiming(m):
    """Initialize weights with Kaiming normal for ASPP blocks"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init_xavier(m):
    """Initialize weights with Xavier uniform for the overall model"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    """ Double Convolution (DoubleConv) Layer:
        (Conv2D => BatchNorm => ReLU) ---> (Conv2D => BatchNorm => ReLU)
        
        If residual=True, then:
         __________________________________ + __________________________________
        |                                                                       |
        x => (Conv2D => BatchNorm => ReLU) ---> (Conv2D => BatchNorm => ReLU) => x'

    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super().__init__()
        self.residual = residual
        
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

        # 1x1 projection if residual is enabled and channels differ
        self.proj = None
        if self.residual and in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x if self.residual else None
            
        x = self.double_conv(x)
        
        if self.residual:
            if self.proj is not None:
                residual = self.proj(residual)
            x = x + residual
        return x

class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18], max_pooling=True):
        super(ASPPBlock, self).__init__()
        self.max_pooling = max_pooling

        if self.max_pooling:
            self.pool = nn.MaxPool2d(2)
 
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.output = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

        # Apply Kaiming initialization to ASPP block
        self.apply(_init_weights_kaiming)

    def forward(self, x):
        if self.max_pooling:
            x = self.pool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

class Down(nn.Module):
    """ Downscaling Module:
        MaxPool2D => DoubleConv => Optional SELayer
    """
    def __init__(self, in_channels, out_channels, use_se=False, use_residuals=True):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, None, use_residuals)

        self.se = SELayer(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        x = self.se(x)
        return x

class Up(nn.Module):
    """ Upscaling Module:
        x => Upsample => Concat with x => Optional SELayer => DoubleConv
    """

    def __init__(self, in_channels, out_channels, bilinear=True, use_se=True, use_residuals=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.se = SELayer(in_channels) if use_se else nn.Identity() # After upsampling, SE layer needs half of the channels
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2 if bilinear else None, use_residuals)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample first (reduces channels)

        # Pad the upsampled tensor to match the skip connection tensor
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)  # Then concatenate
        
        x = self.se(x)  # Then SE block (on reduced channels)
        x = self.conv(x)  # Then apply DoubleConv
        return x

class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 33, bilinear: bool = False, use_se_enc: bool = True, use_se_dec: bool = True, use_aspp_block: bool = True, use_dropout: bool = False, use_residuals: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_se_enc = use_se_enc
        self.use_se_dec = use_se_dec
        self.use_aspp_block = use_aspp_block
        self.use_dropout = use_dropout
        self.use_residuals = use_residuals
        
        # Encoder with Attention block integrated into Downsample Layer
        self.inc = DoubleConv(n_channels, 64, None, self.use_residuals)
        self.am = SELayer(64) if self.use_se_enc else nn.Identity()
        
        self.down1 = Down(64, 128, self.use_se_enc, self.use_residuals)
        self.down2 = Down(128, 256, self.use_se_enc, self.use_residuals)
        self.down3 = Down(256, 512, self.use_se_enc, self.use_residuals)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_se=False, use_residuals=False)  # Bottleneck without Attention & Residual

        # Bottleneck ASPP
        self.aspp_bottleneck = ASPPBlock(1024 // factor, 1024 // factor, dilations=[6, 12, 18], max_pooling=False) if self.use_aspp_block else nn.Identity()

        # Decoder with Attention blocks integrated into Upsample Layer
        self.up1 = Up(1024, 512 // factor, bilinear, self.use_se_dec, self.use_residuals)
        self.up2 = Up(512, 256 // factor, bilinear, self.use_se_dec, self.use_residuals)
        self.up3 = Up(256, 128 // factor, bilinear, self.use_se_dec, self.use_residuals)
        self.up4 = Up(128, 64, bilinear, self.use_se_dec, self.use_residuals)
        
        # Final ASPP Block
        self.aspp_output = ASPPBlock(64, 64, dilations=[6, 12, 18], max_pooling=False) if self.use_aspp_block else nn.Identity()

        # Optional Dropout Layer
        if self.use_dropout:
            self.dropout = nn.Dropout(0.1)
        
        # Final Attention Layer
        self.am1 = SELayer(64) if self.use_se_dec else nn.Identity()

        # Classifier
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

        # Initialize weights
        self.apply(weights_init_xavier) # Apply Xavier uniform initialization to the entire model
        self.apply(init_he)             # Apply He initialization to Conv2D layers

    def forward(self, x):
        # Encoder with skip connections stored
        x1 = self.inc(x)        # 64, H, W
        x1 = self.am(x1)      # Apply Attention layer, if enabled
        x2 = self.down1(x1)     # 128, H/2, W/2
        x3 = self.down2(x2)     # 256, H/4, W/4
        x4 = self.down3(x3)     # 512, H/8, W/8
        x5 = self.down4(x4)     # 1024, H/16, W/16

        # ASPP at bottleneck (Apply only if enabled)
        x5 = self.aspp_bottleneck(x5)  # 1024, H/16, W/16

        # Decoder with skip connections
        x = self.up1(x5, x4)    # 512, H/8, W/8
        x = self.up2(x, x3)     # 256, H/4, W/4
        x = self.up3(x, x2)     # 128, H/2, W/2
        x = self.up4(x, x1)     # 64, H, W

        # ASPP at final output (Apply only if enabled)
        x = self.aspp_output(x)  # 64, H, W

        # Optional Dropout Layer
        if self.use_dropout:
            x = self.dropout(x)
            
        # Apply Attention layer at the end, if enabled
        x = self.am1(x)

        logits = self.classifier(x) # num_classes, H, W
        return logits

if __name__ == '__main__':

    # Create model with proper weight initialization
    in_channels = 1  # Grayscale input
    num_classes = 33
    bilinear = True
    use_se_enc = False
    use_se_dec = False
    use_aspp_block = False
    use_dropout = False
    use_residuals = False
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Initializing UNet with the following parameters:")
    print(f" - Input channels: {in_channels}")
    print(f" - Number of classes: {num_classes}")
    print(f" - Bilinear upsampling: {bilinear}")
    print(f" - Use SE in encoder: {use_se_enc}")
    print(f" - Use SE in decoder: {use_se_dec}")
    print(f" - Use ASPP block: {use_aspp_block}")
    print(f" - Use dropout: {use_dropout}")
    print(f" - Use residuals: {use_residuals}")

    model = UNet(in_channels, num_classes, bilinear, use_se_enc, use_se_dec, use_aspp_block, use_dropout, use_residuals).to(device)

    # Test forward pass
    x = torch.randn((1, 1, 512, 512), device=device)  # Example input tensor (batch_size, channels, height, width)

    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
