from collections import namedtuple
import torch
from torchvision import models as tv
from IPython import embed
from torchvision.models import VGG16_Weights,SqueezeNet1_1_Weights,AlexNet_Weights,ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
import math

def adapt_input_conv(in_chans: int, conv_weight: torch.Tensor) -> torch.Tensor:
    """Adapt pretrained RGB conv weights to different input channels."""
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        repeat = int(math.ceil(in_chans / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= (3 / float(in_chans))
    
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, in_chans=3):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).features

        # Adapt first conv if needed
        if in_chans != 3 and pretrained:
            first_conv = pretrained_features[0]
            if isinstance(first_conv, torch.nn.Conv2d):
                with torch.no_grad():
                    adapted_weight = adapt_input_conv(in_chans, first_conv.weight.data)
                new_conv = torch.nn.Conv2d(
                    in_chans, first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                new_conv.weight.data = adapted_weight
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()
                pretrained_features[0] = new_conv

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, in_chans=3):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(weights=AlexNet_Weights.DEFAULT).features

        # Adapt first conv if needed
        if in_chans != 3 and pretrained:
            first_conv = alexnet_pretrained_features[0]
            if isinstance(first_conv, torch.nn.Conv2d):
                with torch.no_grad():
                    adapted_weight = adapt_input_conv(in_chans, first_conv.weight.data)
                new_conv = torch.nn.Conv2d(
                    in_chans, first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                new_conv.weight.data = adapted_weight
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()
                alexnet_pretrained_features[0] = new_conv

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, in_chans=3):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(weights=VGG16_Weights.DEFAULT).features

        # Adapt first conv if needed
        if in_chans != 3 and pretrained:
            first_conv = vgg_pretrained_features[0]
            if isinstance(first_conv, torch.nn.Conv2d):
                with torch.no_grad():
                    adapted_weight = adapt_input_conv(in_chans, first_conv.weight.data)
                new_conv = torch.nn.Conv2d(
                    in_chans, first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                new_conv.weight.data = adapted_weight
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()
                vgg_pretrained_features[0] = new_conv

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out



class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18, in_chans=3):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif(num==34):
            self.net = tv.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif(num==50):
            self.net = tv.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif(num==101):
            self.net = tv.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif(num==152):
            self.net = tv.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out
