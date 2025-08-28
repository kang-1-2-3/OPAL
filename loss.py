import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

    
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=10):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, osm_descriptors, pc_descriptors):

        similarity_matrix = F.cosine_similarity(
            osm_descriptors.unsqueeze(1),
            pc_descriptors.unsqueeze(0),
            dim=2
        )

        positive_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        negative_mask = ~positive_mask

        sp = similarity_matrix[positive_mask].view(similarity_matrix.size(0), -1)
        sn = similarity_matrix[negative_mask].view(similarity_matrix.size(0), -1)

        alpha_p = torch.clamp_min(-sp + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sn + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        pos_term = torch.sum(torch.exp(-self.gamma * alpha_p * (sp - delta_p)), dim=1)
        neg_term = torch.sum(torch.exp(self.gamma * alpha_n * (sn - delta_n)), dim=1)

        loss = torch.log1p(pos_term * neg_term)

        return loss.mean()
    


