from typing import Optional
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        if self.dim == 2:
            return gem_2d(x, p=self.p, eps=self.eps)
        elif self.dim == 3:
            return gem_3d(x, p=self.p, eps=self.eps)
        elif self.dim == 1:
            return gem_1d(x, p=self.p, eps=self.eps)

def tf_efficientnet_b6_ns(pretrained=True, stride_down=False, **kwargs):
    model = timm.models.tf_efficientnet_b6_ns(pretrained=pretrained, **kwargs)
    if stride_down:
        print('Reducing stride from (2,2) to (1,1) ...')
        model.conv_stem.stride = (1,1)
        #model.blocks[1][0].conv_dw.stride = (1,1)
    dim_feats = model.classifier.in_features
    # model.classifier = pretrainedmodels.utils.Identity()

    # use identity
    model.classifier = nn.Identity()
    return model, dim_feats


class Net2D(nn.Module):


    def __init__(self,
                 backbone: str = 'tf_efficientnet_b6_ns',
                 pretrained: Optional[bool] = None,
                 num_classes: int = 8,
                 multisample_dropout=True,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem'):
        super().__init__()
        self.backbone, dim_feats = tf_efficientnet_b6_ns(pretrained=pretrained, **backbone_params)

        # Change pooling layer, if specified
        if pool == 'gem':
            setattr(self.backbone, 'global_pool', GeM())

        self.multisample_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward_base(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        if self.multisample_dropout:
            x = torch.mean(
                torch.stack(
                    [self.fc(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.fc(self.dropout(features))

        x = x if self.fc.out_features > 1 else x[:,0]

        return x


    def forward(self, x):
        square_tta = False
        if x.ndim == 5 and x.size(0) == 1:
            # shape = (1, num_crops, C, H, W)
            x = x.squeeze(0)
            square_tta = True
        x = self.forward_base(x)

        if square_tta:
            # shape = (N, num_classes)
            x = x.mean(0).unsqueeze(0)

        return x
