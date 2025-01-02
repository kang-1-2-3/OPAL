import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=5):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, osm_descriptors, pc_descriptors):
        """
        calculate circle loss
        :param osm_descriptors: OSM global descriptors, shape (batch_size, descriptor_dim)
        :param pc_descriptors: Point Cloud global descriptors, shape (batch_size, descriptor_dim)
        :return: circle loss value 
        """
        # L2 normalize the descriptors
        osm_descriptors = F.normalize(osm_descriptors, p=2, dim=1)
        pc_descriptors = F.normalize(pc_descriptors, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(osm_descriptors, pc_descriptors.T)
        
        # Create positive and negative masks
        positive_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        negative_mask = 1 - positive_mask
        
        # Split similarities for positive and negative pairs
        sp = similarity_matrix[positive_mask.bool()].view(similarity_matrix.size(0), -1)
        sn = similarity_matrix[negative_mask.bool()].view(similarity_matrix.size(0), -1)

        # Calculate optimal target for positive and negative pairs
        alpha_p = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * alpha_p * (sp - delta_p)
        logit_n = self.gamma * alpha_n * (sn - delta_n)

        # Compute log sum of positive and negative pairs
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        
        return loss.mean()

class CircleLossV2(nn.Module):

    def __init__(self, m=0.25, gamma=10):
        super(CircleLossV2, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, osm_descriptors, pc_descriptors):
        """
        calculate circle loss with full implementation
        :param osm_descriptors: OSM global descriptors, shape (batch_size, descriptor_dim)
        :param pc_descriptors: Point Cloud global descriptors, shape (batch_size, descriptor_dim)
        :return: circle loss value 
        """
        # 直接使用cosine_similarity计算相似度矩阵
        similarity_matrix = F.cosine_similarity(
            osm_descriptors.unsqueeze(1), 
            pc_descriptors.unsqueeze(0), 
            dim=2
        )
        
        positive_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        negative_mask = 1 - positive_mask
        
        # Split similarities
        sp = similarity_matrix[positive_mask.bool()].view(similarity_matrix.size(0), -1)
        sn = similarity_matrix[negative_mask.bool()].view(similarity_matrix.size(0), -1)

        # Calculate adaptive weights
        alpha_p = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * alpha_p * (sp - delta_p)
        logit_n = self.gamma * alpha_n * (sn - delta_n)

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        
        return loss.mean()

def compute_feat_dists(osm_global_feature, pc_global_feature):

    feat_dists = torch.cdist(osm_global_feature, pc_global_feature, p=2)  
    return feat_dists

def compute_pos_neg_masks(feat_dists):

    pos_masks = torch.eye(feat_dists.size(0), device=feat_dists.device).bool()
    neg_masks = ~torch.eye(feat_dists.size(0), device=feat_dists.device).bool()
    return pos_masks, neg_masks

def circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss

class CircleLossV3(nn.Module):
    def __init__(
        self,
        pos_margin=0.1,
        neg_margin=1.4,
        pos_optimal=0.1,
        neg_optimal=1.4,
        log_scale=40,
    ):
        super(CircleLossV3, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, osm_global_feature, pc_global_feature):
        feat_dists = compute_feat_dists(osm_global_feature, pc_global_feature)

        pos_masks, neg_masks = compute_pos_neg_masks(feat_dists)

        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )

class CircleLossV4(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, log_scale=10):
        super(CircleLossV4, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.log_scale = log_scale

    def forward(self, img_features, pc_features):
        
        dists = 1 - torch.sum(img_features.unsqueeze(-1) * pc_features.unsqueeze(-2), dim=0)
        mask = torch.eye(dists.size(0), device=img_features.device)
        pos_mask = mask
        neg_mask = 1 - mask

        pos = dists - 1e5 * neg_mask
        pos_weight = (pos - self.pos_margin).detach()
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

        lse_positive_row = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-1)
        lse_positive_col = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-2)

        neg = dists + 1e5 * pos_mask
        neg_weight = (self.neg_margin - neg).detach()
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)

        lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-1)
        lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-2)

        loss_col = F.softplus(lse_positive_row + lse_negative_row) / self.log_scale
        loss_row = F.softplus(lse_positive_col + lse_negative_col) / self.log_scale
        loss = loss_col + loss_row

        return torch.mean(loss)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            temperature (float, optional): Scaling factor for similarity logits. Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, osm_feature, pc_feature):
        # Normalize the input features
        # osm_feature = osm_feature / torch.norm(osm_feature, dim=1, keepdim=True)
        # pc_feature = pc_feature / torch.norm(pc_feature, dim=1, keepdim=True)

        # Compute the similarity matrix
        similarity = torch.mm(osm_feature, pc_feature.transpose(1, 0).contiguous())

        # Extract positive samples (diagonal elements)
        positives = torch.diag(similarity)

        # Compute numerator and denominator for contrastive loss
        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature)

        # Compute loss for both directions
        all_losses = - torch.log(numerator / torch.sum(denominator, dim=0)) \
                     - torch.log(numerator / torch.sum(denominator, dim=1))

        # Average the loss over all samples
        loss = torch.mean(all_losses)

        return loss