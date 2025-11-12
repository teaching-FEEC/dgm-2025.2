from torch import nn
import torch

class ISIC2019Model(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = torch.jit.load(model_path)

    def forward(self, x):
        x = self.model(x)
        return x

        # ISIC2019 has 8 classes, with 0 being melanoma, 1 being nevus, and 2-7 being other classes.
        # the output of the model is [batch_size, 8]
        # We want to convert this to a binary classification problem: melanoma (0) vs non-melanoma (1)
        # if self.task == "melanoma_vs_non_melanoma":
        #     melanoma_logits = x[:, 0]  # logits for melanoma
        #     non_melanoma_logits = torch.logsumexp(x[:, 1:], dim=1)  # logits for non-melanoma
        #     binary_logits = torch.stack([melanoma_logits, non_melanoma_logits], dim=1)
        # elif self.task == "melanoma_vs_dysplastic_nevi":
        #     melanoma_logits = x[:, 0]
        #     dysplastic_logits = x[:, 1]
        #     binary_logits = torch.stack([melanoma_logits, dysplastic_logits], dim=1)
        # elif self.task == "all_classes":
        #     binary_logits = x  # return all 8 classes as is
        # else:
        #     raise ValueError(f"Unknown task: {self.task}")
        # return binary_logits
