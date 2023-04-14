import numpy as np
import torch
import torch.nn as tn
import torch.nn.functional as tnf


class ContrastiveLoss(tn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.from_numpy(np.array(np.linalg.norm(output1 - output2))).view(1)
        euclidean_distance = torch.pow(euclidean_distance, 2)
        loss_contrastive = torch.mean(
                    (1 - label) * euclidean_distance + label * torch.pow( torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
                )
        return loss_contrastive