import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from random import randint

seq = nn.Sequential


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except Exception:
            pass
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


# Implementation of the SLE module, a variant of the Squeeze-and-Excitation block that can take 2 feature maps
class SEBlock(nn.Module):
    def __init__(
        self,
        ch_in,   # Number of channels in the small input feature map
        ch_out,  # Number of channels for the output gating feature map
    ):
        super().__init__()
        # Consists of adaptive average pooling, a convolution, Swish activation, another convolution, and Sigmoid activation
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        # First apply a series of ops to feat_small
        # to obtain a tensor of shape (batch_size, ch_out, 1, 1)
        # where each element represents the importance weight for that channel.
        # Then apply these weights to the larger feature map feat_big via element-wise multiplication,
        # i.e., each channel of the large feature is scaled by its corresponding weight.
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False), batchNorm2d(channel * 2), GLU()
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    # Wrap one or more modules in sequence so data flows through them in the defined order
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2),
        GLU(),
    )
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2),
        GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2),
        GLU(),
    )
    return block


# Define the original Generator class
class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super().__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input, mask):
        mask = mask.view(mask.size(0), -1).float()
        input = torch.cat([input, mask], dim=-1)

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        # Apply the SLE modules
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            im_256 = torch.tanh(self.to_big(feat_256))
            im_128 = torch.tanh(self.to_128(feat_128))
            return [im_256, im_128]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))

        if self.im_size == 512:
            im_512 = torch.tanh(self.to_big(feat_512))
            im_128 = torch.tanh(self.to_128(feat_128))
            return [im_512, im_128]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


# Define a simple downsampling block
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


# Define a composite downsampling block
class DownBlockComp(nn.Module):
    def __init__(
        self,
        in_planes,   # Number of channels in the input feature map
        out_planes,  # Number of channels in the output feature map
    ):
        # Initialization: define layers and parameters for this class
        super().__init__()
        # Standard convolutional downsampling path
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            # Use a 4x4 kernel, stride 2, padding 1 for downsampling
            # Converts from in_planes to out_planes and halves spatial resolution
            batchNorm2d(out_planes),
            # Batch normalization to accelerate training and stabilize the model
            nn.LeakyReLU(0.2, inplace=True),
            # Apply LeakyReLU with negative slope 0.2 and in-place to save memory
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            # 3x3 kernel, stride 1, padding 1: keep spatial size, further transform features
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2),
        )
        # Direct path: average pooling for downsampling followed by 1x1 conv to adjust channels
        self.direct = nn.Sequential(
            # Use average pooling to halve spatial resolution
            nn.AvgPool2d(2, 2),
            # 1x1 convolution to change channels from in_planes to out_planes without changing spatial size
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            # Batch normalization
            batchNorm2d(out_planes),
            # Apply LeakyReLU with negative slope 0.2 and in-place to save memory
            nn.LeakyReLU(0.2),
        )

    def forward(self, feat):
        # Define the forward path through the network as the average of the two paths above
        return (self.main(feat) + self.direct(feat)) / 2


# Define the original Discriminator class
class Discriminator(nn.Module):
    # In PyTorch's nn.Module, __call__ is overloaded to invoke forward() and add extra features
    def __init__(
        self,
        ndf=64,
        nc=3,  # Default 3 for color images
        im_size=512,
    ):
        super().__init__()
        self.ndf = ndf  # Number of feature channels, default 64
        # nc is the number of image channels
        self.im_size = im_size  # Image size

        # Coefficients for rules that set feature-map channel counts per image size
        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}  # Dict of channel counts per spatial size; key is size, value is channels
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)  # For 256 resolution, equals 0.5*64

        # Downsampling strategy for large images; varies with input image size
        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),  # stride 2
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),  # stride 2
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),  # stride 2
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),  # stride 1
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Downsampling modules: channels increase as spatial size decreases
        self.down_4 = DownBlockComp(nfc[512], nfc[256])   # Stage I downsampling; 16 32
        self.down_8 = DownBlockComp(nfc[256], nfc[128])   # Stage II downsampling; 32 64
        self.down_16 = DownBlockComp(nfc[128], nfc[64])   # Stage III downsampling; 64 128
        self.down_32 = DownBlockComp(nfc[64], nfc[32])    # Stage IV downsampling; 128 256
        self.down_64 = DownBlockComp(nfc[32], nfc[16])    # Stage V downsampling; 256 512

        # Receptive Field head for final prediction on the original-sized image path
        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False),
        )

        self.se_2_16 = SEBlock(nfc[512], nfc[64])   # SLE module I; 16 128
        self.se_4_32 = SEBlock(nfc[256], nfc[32])   # SLE module II; 32 256
        self.se_8_64 = SEBlock(nfc[128], nfc[16])   # SLE module III; 64 512

        # Downsampling strategy for small images
        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        )

        # Receptive Field head for prediction on the reduced-size image path
        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        # Three decoders that reconstruct images from features of different sizes
        # Decoder for the original-sized image
        self.decoder_big = SimpleDecoder(nfc[16], nc)   # 512 -> 3
        # Decoder for partial image patches
        self.decoder_part = SimpleDecoder(nfc[32], nc)  # 256 -> 3
        # Decoder for the small image
        self.decoder_small = SimpleDecoder(nfc[32], nc) # 256 -> 3

    # Forward pass is invoked directly; defines the discriminator's execution flow
    # imgs is a single image or a list of images
    def forward(self, imgs, label, mask, part=None):
        imgs = torch.cat([imgs.view(imgs.size(0), -1), mask.view(mask.size(0), -1).float()], dim=-1) #new
        # Preprocess by interpolation; ensure a list with original and reduced images
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        # Process the original-sized image
        feat_2 = self.down_from_big(imgs[0])  # directly process the original image (possibly augmented)
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)            # cascaded downsampling
        feat_16 = self.se_2_16(feat_2, feat_16)   # SLE

        feat_32 = self.down_32(feat_16)           # cascaded downsampling
        feat_32 = self.se_4_32(feat_4, feat_32)   # SLE

        # Final feature map
        feat_last = self.down_64(feat_32)         # cascaded downsampling
        feat_last = self.se_8_64(feat_8, feat_last)  # SLE

        # Discrimination score for the original-sized path
        rf_0 = self.rf_big(feat_last).view(-1)

        # Process the reduced-size image
        feat_small = self.down_from_small(imgs[1])
        # rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        # Discrimination score for the reduced-size path
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == "real":  # label is an input flag indicating real images
            rec_img_big = self.decoder_big(feat_last)    # reconstruction for the original image
            # size --> (batch size, 3, 256, 256)

            rec_img_small = self.decoder_small(feat_small)  # reconstruction for the reduced image
            # size --> (batch size, 3, 128, 128)

            # part is a random integer in [0, 4)
            assert part is not None
            rec_img_part = None
            # size --> (batch size, 3, 32, 32)
            # Choose different spatial quadrants based on the random part and reconstruct that patch
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8].contiguous())
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:].contiguous())
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8].contiguous())
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:].contiguous())

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]
            # For real images, also return reconstructions

        return torch.cat([rf_0, rf_1])  # size --> (batch size*2,)


# Define a simple decoder for use in the discriminator
class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nfc_in=64, nc=3):
        # Default channels for feature maps; actual input channels depend on which decoder is used
        # Image channel count
        super().__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}  # Dict of feature-map channel counts; key is size, value is channels
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        # Define an upsampling module
        def upBlock(
            in_planes,   # Number of channels in the input feature map
            out_planes,  # Number of channels in the output feature map
        ):
            #
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                # in_channels=in_planes, input channels; out_channels=out_planes*2, output channels
                # kernel_size=3, use a 3x3 kernel
                # stride=1, unit stride
                # padding=1, keep spatial size unchanged
                batchNorm2d(out_planes * 2),
                GLU(),  # Split channels into two halves: one as gate, the other linearly transformed then multiplied by the gate
            )
            return block

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            upBlock(nfc_in, nfc[16]),   # Upsample based on the given input feature channels
            upBlock(nfc[16], nfc[32]),
            upBlock(nfc[32], nfc[64]),
            upBlock(nfc[64], nfc[128]),
            conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h - size - 1)
    cw = randint(0, w - size - 1)
    return image[:, :, ch : ch + size, cw : cw + size]


# Define the texture discriminator class; this class is not actually used in practice
class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super().__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 8, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        )
        self.rf_small = nn.Sequential(conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)

        def forward(self, img, label):
            img = random_crop(img, size=128)

            feat_small = self.down_from_small(img)
            rf = self.rf_small(feat_small).view(-1)

            if label == "real":
                rec_img_small = self.decoder_small(feat_small)

                return rf, rec_img_small, img

            return rf
