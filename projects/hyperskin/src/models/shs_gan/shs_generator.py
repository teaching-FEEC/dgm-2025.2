import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small building blocks ----------

class DoubleConv(nn.Module):
    """
    (Conv2d -> LeakyReLU) x 2 with padding=1 to keep spatial size.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    Upsample via ConvTranspose2d (stride=2) then DoubleConv.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # after up: channels = in_ch // 2; we will cat with skip (same channels) ->  in_ch
        self.double = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)                      # (B, in_ch//2, H*2, W*2)
        x = torch.cat([x, skip], dim=1)     # channel concat
        return self.double(x)

# ---------- Conditional U-Net Generator ----------

class ConditionalUNetGenerator(nn.Module):
    """
    U-Net generator with class conditioning and per-pixel noise maps.

    Inputs:
      - x_in:     (B, Cin, 256, 256)      -> input image/canvas (set Cin=3 for RGB; can be zeros)
      - y:        (B,) long               -> class indices in [0, K-1]
      - nz_maps:  integer                  -> number of per-pixel noise channels
      - K:        integer                  -> number of classes

    Conditioning:
      - y is converted to one-hot and broadcast to HxW -> (B, K, 256, 256)
      - z_maps ~ N(0,1) -> (B, nz_maps, 256, 256)
      - U-Net input is cat([x_in, y_maps, z_maps], dim=1)

    Architecture:
      Enc:  (Cin+K+nz) -> 64 -> 128 -> 256 -> 512  with MaxPool2d between
      Dec:  512 -> 256 -> 128 -> 64 with skip connections
      Out:  64 -> Cout (16), then Tanh
    """
    def __init__(self, Cin=3, Cout=16, K=4, nz_maps=4, base=64):
        super().__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.K = K
        self.nz_maps = nz_maps

        in_ch = Cin + K + nz_maps

        # Encoder
        self.enc1 = DoubleConv(in_ch, base)          # 64 x 256 x 256
        self.pool1 = nn.MaxPool2d(2)                 # -> 128x128

        self.enc2 = DoubleConv(base, base*2)         # 128 x 128 x 128
        self.pool2 = nn.MaxPool2d(2)                 # -> 64x64

        self.enc3 = DoubleConv(base*2, base*4)       # 256 x 64 x 64
        self.pool3 = nn.MaxPool2d(2)                 # -> 32x32

        self.enc4 = DoubleConv(base*4, base*8)       # 512 x 32 x 32  (bottleneck)

        # Decoder (upsample + skip)
        self.up3 = Up(base*8, base*4)                # 32->64, out 256 ch
        self.up2 = Up(base*4, base*2)                # 64->128, out 128 ch
        self.up1 = Up(base*2, base)                  # 128->256, out 64 ch

        # Output head
        self.out = nn.Conv2d(base, Cout, kernel_size=1)

    def forward(self, x_in, y):
        B, _, H, W = x_in.shape
        assert H == 256 and W == 256, "This U-Net is set for 256x256. Adjust pooling/upsample for other sizes."

        # Class maps 
        y_oh = F.one_hot(y, num_classes=self.K).float()           # (B, K)
        y_maps = y_oh[:, :, None, None].expand(-1, -1, H, W)      # (B, K, 256, 256)

        z_maps = torch.randn(B, self.nz_maps, H, W, device=x_in.device)

        x0 = torch.cat([x_in, y_maps, z_maps], dim=1)             # (B, Cin+K+nz, 256, 256)

        e1 = self.enc1(x0)                    # (B, 64, 256, 256)
        p1 = self.pool1(e1)                   # (B, 64, 128, 128)

        e2 = self.enc2(p1)                    # (B, 128, 128, 128)
        p2 = self.pool2(e2)                   # (B, 128, 64, 64)

        e3 = self.enc3(p2)                    # (B, 256, 64, 64)
        p3 = self.pool3(e3)                   # (B, 256, 32, 32)

        b  = self.enc4(p3)                    # (B, 512, 32, 32)

       
        d3 = self.up3(b,  e3)                 # (B, 256, 64, 64)
        d2 = self.up2(d3, e2)                 # (B, 128, 128, 128)
        d1 = self.up1(d2, e1)                 # (B, 64, 256, 256)

        out = torch.tanh(self.out(d1))        # (B, 29, 256, 256)
        return out
