import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic3D(nn.Module):
    def __init__(self, in_channels=16, fft_arm=True):  
        """
        Critic for SHS-GAN.
        Input: Hyperspectral cube [B, C, H, W]
        Output: Scalar WGAN score
        """
        super(Critic3D, self).__init__()
        self.fft_arm = fft_arm

        # Spatial Conv Arm
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5,3,3), stride=1, padding=(2,1,1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        #FFT Conv Arm 
        if self.fft_arm:
            self.fft_conv = nn.Sequential(
                nn.Conv3d(2, 32, kernel_size=(5,3,3), stride=1, padding=(2,1,1)),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        #Fully Connected Head 
 
        fc_in_dim = 128*2 if self.fft_arm else 128
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)  # scalar critic score
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Spatial Arm
        x_spatial = x.unsqueeze(1)   # [B,1,C,H,W]
        out_spatial = self.spatial_conv(x_spatial)
        out_spatial = F.adaptive_avg_pool3d(out_spatial, 1).view(B, -1) 

        out = out_spatial

        # FFT Arm 
        #avaliar melhor estes passos
        if self.fft_arm:
            x_fft = torch.fft.fft(x, dim=1)       
            x_fft = torch.view_as_real(x_fft)     
            x_fft = x_fft.permute(0,4,1,2,3)     
            out_fft = self.fft_conv(x_fft)
            out_fft = F.adaptive_avg_pool3d(out_fft, 1).view(B, -1) 

          
            out = torch.cat([out_spatial, out_fft], dim=1)  

        
        return self.fc(out)  # scalar per sample
