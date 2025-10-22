import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, base_filters=64, dropout_prob=0.5):
        super(Generator, self).__init__()
        
        # Encoder 
        self.enc1 = nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1)  
        self.enc2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=4, stride=2, padding=1)  
        self.enc3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=4, stride=2, padding=1)  
        self.enc4 = nn.Conv2d(base_filters*4, base_filters*8, kernel_size=4, stride=2, padding=1)  

        # Bottleneck
        self.bottleneck = nn.Conv2d(base_filters*8, base_filters*8, kernel_size=4, stride=2, padding=1)  

        # Decoder 
        self.dec4 = nn.ConvTranspose2d(base_filters*8, base_filters*8, kernel_size=4, stride=2, padding=1)  
        self.dec3 = nn.ConvTranspose2d(base_filters*8*2, base_filters*4, kernel_size=4, stride=2, padding=1)  
        self.dec2 = nn.ConvTranspose2d(base_filters*4*2, base_filters*2, kernel_size=4, stride=2, padding=1)  
        self.dec1 = nn.ConvTranspose2d(base_filters*2*2, base_filters, kernel_size=4, stride=2, padding=1)  

        # Output
        self.final = nn.ConvTranspose2d(base_filters*2, out_channels, kernel_size=4, stride=2, padding=1)  

        # Normalization 
        self.bn1 = nn.BatchNorm2d(base_filters*2)
        self.bn2 = nn.BatchNorm2d(base_filters*4)
        self.bn3 = nn.BatchNorm2d(base_filters*8)
        self.bn4 = nn.BatchNorm2d(base_filters*8)  

        #  Dropout 
        self.drop = nn.Dropout(dropout_prob)  

    def forward(self, x):
        # Encoder 
        e1 = F.leaky_relu(self.enc1(x), 0.2)                    
        e2 = F.leaky_relu(self.bn1(self.enc2(e1)), 0.2)            
        e3 = F.leaky_relu(self.bn2(self.enc3(e2)), 0.2)           
        e4 = F.leaky_relu(self.bn3(self.enc4(e3)), 0.2)           
        e4 = self.bn4(e4)  

        # Bottleneck
        b = F.relu(self.bottleneck(e4))                          

        # Decoder
        d4 = F.relu(self.dec4(b))                              
        d4 = self.drop(d4)     
        d4 = torch.cat([d4, e4], dim=1)                            

        d3 = F.relu(self.dec3(d4))                                 
        d3 = self.drop(d3) 
        d3 = torch.cat([d3, e3], dim=1)                            

        d2 = F.relu(self.dec2(d3))                                 
        d2 = self.drop(d2)    
        d2 = torch.cat([d2, e2], dim=1)                            

        d1 = F.relu(self.dec1(d2))                                 
        d1 = torch.cat([d1, e1], dim=1)                            
        out = torch.tanh(self.final(d1))                           # (B, out_channels, 224, 224)

        return out
