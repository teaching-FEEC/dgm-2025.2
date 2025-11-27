import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.fastgan.conditional_fastgan import SEBlock
from src.models.fastgan.fastgan import GLU, DownBlock, InitLayer, NoiseInjection, SimpleDecoder, batchNorm2d, conv2d

# ==================== SPADE Components ====================


class SPADE(nn.Module):
    """
    SPADE normalization layer that conditions on a semantic map/conditioning input
    """

    def __init__(self, norm_nc, label_nc, ks=3, norm_type="batch"):
        super().__init__()

        # Choose normalization type
        if norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f"{norm_type} not recognized in SPADE")

        # Hidden dimension for MLP
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Normalize activations
        normalized = self.param_free_norm(x)

        # Produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    """
    ResNet block with SPADE normalization
    """

    def __init__(self, fin, fout, label_nc, use_spectral_norm=True):
        super().__init__()

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # Create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # Apply spectral norm
        if use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # SPADE normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


# ==================== Enhanced Blocks with SPADE ====================


def UpBlockSPADE(in_planes, out_planes, label_nc):
    """Upsampling block with SPADE normalization"""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)
            self.spade = SPADE(out_planes * 2, label_nc)
            self.glu = GLU()

        def forward(self, x, seg):
            x = self.upsample(x)
            x = self.conv(x)
            x = self.spade(x, seg)
            x = self.glu(x)
            return x

    return Block()


def UpBlockCompSPADE(in_planes, out_planes, label_nc):
    """Composite upsampling block with SPADE"""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv1 = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)
            self.noise1 = NoiseInjection()
            self.spade1 = SPADE(out_planes * 2, label_nc)
            self.glu1 = GLU()

            self.conv2 = conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False)
            self.noise2 = NoiseInjection()
            self.spade2 = SPADE(out_planes * 2, label_nc)
            self.glu2 = GLU()

        def forward(self, x, seg):
            x = self.upsample(x)
            x = self.conv1(x)
            x = self.noise1(x)
            x = self.spade1(x, seg)
            x = self.glu1(x)

            x = self.conv2(x)
            x = self.noise2(x)
            x = self.spade2(x, seg)
            x = self.glu2(x)
            return x

    return Block()


# ==================== Enhanced Generator with SPADE ====================


class GeneratorSPADE(nn.Module):
    """
    Enhanced FastGAN Generator with SPADE blocks.
    Can use noise vector or class labels as conditioning.
    """

    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, label_nc=None):
        super().__init__()

        # If label_nc not specified, use nz (noise) as conditioning
        if label_nc is None:
            label_nc = nz

        self.label_nc = label_nc

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}

        self.im_size = im_size

        # Initial layer
        self.init = InitLayer(nz, channel=nfc[4])

        # Upsampling blocks with SPADE
        self.feat_8 = UpBlockCompSPADE(nfc[4], nfc[8], label_nc)
        self.feat_16 = UpBlockSPADE(nfc[8], nfc[16], label_nc)
        self.feat_32 = UpBlockCompSPADE(nfc[16], nfc[32], label_nc)
        self.feat_64 = UpBlockSPADE(nfc[32], nfc[64], label_nc)
        self.feat_128 = UpBlockCompSPADE(nfc[64], nfc[128], label_nc)
        self.feat_256 = UpBlockSPADE(nfc[128], nfc[256], label_nc)

        # SE blocks
        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        # Output layers
        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockCompSPADE(nfc[256], nfc[512], label_nc)
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlockSPADE(nfc[512], nfc[1024], label_nc)

    def forward(self, input, seg=None):
        """
        Args:
            input: noise vector [B, nz]
            seg: conditioning/segmentation map [B, label_nc, H, W]
                 If None, uses reshaped input as conditioning
        """
        if seg is None:
            # Use noise as conditioning
            seg = input.view(input.shape[0], -1, 1, 1)

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4, seg)
        feat_16 = self.feat_16(feat_8, seg)
        feat_32 = self.feat_32(feat_16, seg)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, seg))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64, seg))
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128, seg))

        if self.im_size == 256:
            im_256 = torch.tanh(self.to_big(feat_256))
            im_128 = torch.tanh(self.to_128(feat_128))
            return [im_256, im_128]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256, seg))

        if self.im_size == 512:
            im_512 = torch.tanh(self.to_big(feat_512))
            im_128 = torch.tanh(self.to_128(feat_128))
            return [im_512, im_128]

        feat_1024 = self.feat_1024(feat_512, seg)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


# ==================== Enhanced Discriminator with SPADE ====================


class DownBlockSPADE(nn.Module):
    """Downsampling block with SPADE"""

    def __init__(self, in_planes, out_planes, label_nc):
        super().__init__()
        self.conv = conv2d(in_planes, out_planes, 4, 2, 1, bias=False)
        self.spade = SPADE(out_planes, label_nc)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, feat, seg):
        feat = self.conv(feat)
        feat = self.spade(feat, seg)
        feat = self.activation(feat)
        return feat


class DownBlockCompSPADE(nn.Module):
    """Composite downsampling block with SPADE"""

    def __init__(self, in_planes, out_planes, label_nc):
        super().__init__()

        # Main path
        self.conv1 = conv2d(in_planes, out_planes, 4, 2, 1, bias=False)
        self.spade1 = SPADE(out_planes, label_nc)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.spade2 = SPADE(out_planes, label_nc)
        self.act2 = nn.LeakyReLU(0.2)

        # Direct path
        self.pool = nn.AvgPool2d(2, 2)
        self.conv_direct = conv2d(in_planes, out_planes, 1, 1, 0, bias=False)
        self.spade_direct = SPADE(out_planes, label_nc)
        self.act_direct = nn.LeakyReLU(0.2)

    def forward(self, feat, seg):
        # Main path
        main = self.conv1(feat)
        main = self.spade1(main, seg)
        main = self.act1(main)
        main = self.conv2(main)
        main = self.spade2(main, seg)
        main = self.act2(main)

        # Direct path
        direct = self.pool(feat)
        direct = self.conv_direct(direct)
        direct = self.spade_direct(direct, seg)
        direct = self.act_direct(direct)

        return (main + direct) / 2


class DiscriminatorSPADE(nn.Module):
    """
    Enhanced Discriminator with SPADE blocks
    """

    def __init__(self, ndf=64, nc=3, im_size=512, label_nc=None):
        super().__init__()
        self.ndf = ndf
        self.im_size = im_size

        # Use image channels as conditioning if not specified
        if label_nc is None:
            label_nc = nc

        self.label_nc = label_nc

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {k: int(v * ndf) for k, v in nfc_multi.items()}

        # Initial downsampling
        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Downsampling with SPADE
        self.down_4 = DownBlockCompSPADE(nfc[512], nfc[256], label_nc)
        self.down_8 = DownBlockCompSPADE(nfc[256], nfc[128], label_nc)
        self.down_16 = DownBlockCompSPADE(nfc[128], nfc[64], label_nc)
        self.down_32 = DownBlockCompSPADE(nfc[64], nfc[32], label_nc)
        self.down_64 = DownBlockCompSPADE(nfc[32], nfc[16], label_nc)

        # Receptive field for big image
        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False),
        )

        # SE blocks
        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        # Small image path
        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        # Decoders
        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, seg=None, part=None):
        """
        Args:
            imgs: input images
            label: 'real' or 'fake'
            seg: conditioning map (if None, uses input image)
            part: random crop location for part reconstruction
        """
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        # Use image as conditioning if seg not provided
        if seg is None:
            seg = imgs[0]

        # Process big image with SPADE conditioning
        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2, seg)
        feat_8 = self.down_8(feat_4, seg)

        feat_16 = self.down_16(feat_8, seg)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16, seg)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32, seg)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf_0 = self.rf_big(feat_last).view(-1)

        # Process small image
        feat_small = self.down_from_small(imgs[1])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == "real":
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8].contiguous())
            elif part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:].contiguous())
            elif part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8].contiguous())
            elif part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:].contiguous())

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])
