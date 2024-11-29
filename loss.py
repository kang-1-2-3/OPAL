import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, osm_descriptors, pc_descriptors):
        """
        calculate circle loss
        :param osm_descriptors: OSM global descriptors, shape (batch_size, descriptor_dim)
        :param pc_descriptors: Point Cloud global descriptors, shape (batch_size, descriptor_dim)
        :return: circle loss value 
        """
        similarity_matrix = torch.matmul(osm_descriptors, pc_descriptors.T)
        positive_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        negative_mask = 1 - positive_mask
        
        positive_similarity = similarity_matrix * positive_mask
        negative_similarity = similarity_matrix * negative_mask
        
        positive_loss = F.relu(-positive_similarity + 1 + self.margin)
        negative_loss = F.relu(negative_similarity + self.margin)
        
        loss = torch.logsumexp(self.gamma * positive_loss, dim=1) + torch.logsumexp(self.gamma * negative_loss, dim=1)
        loss = loss.mean()
        
        return loss
