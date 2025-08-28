# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
from einops import rearrange, einsum
import torch.nn.functional as F
import math

class linear_sigmoid(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels,1,bias = True)
        self.norm = nn.LayerNorm(1)
        self.act = nn.Sigmoid()
    
    def forward(self,x_l,x_h):
        """
        x_l : low-level feature, [B,R_l,C_l]
        x_H : high-level feature, [B,R_h,C_h]
        """
        weight = self.act(self.norm(self.linear(x_h))) # [B,R_l,1]
        weight = F.interpolate(weight.permute(0,2,1),size = x_l.shape[1],mode = "linear").permute(0,2,1).contiguous() # [B,R_l,1]
        
        return x_l * weight

class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, n_dim: int = 1, d_model: int = 64, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class Channel_Att(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Channel_Att, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,1)
        self.norm = nn.BatchNorm2d(out_channels) 
        self.act = nn.ReLU(inplace=True)
    
    def forward(self,x,dim = (2,3)):
        x = x.mean(dim = dim, keepdim = True)
        return self.act(self.norm(self.conv(x))).squeeze().permute(0,2,1) # return [B,N,C] as axial feature

class MLP(nn.Module):
    def __init__(self,in_dim,hid_dim, out_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.dwconv = nn.Conv2d(hid_dim, hid_dim, 3, 1, 1, groups = hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self,x,h,w):
        """
        x: [B,N,C]
        """
        # h,w = x.shape[-2:]
        # x = rearrange(x, 'b c h w -> b (hw) c')
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w',h = h, w = w)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.fc2(x)
        return x


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()

class BoAQ_pe_Block(nn.Module):
    """
    bag of queries (Axial-enhanced version) with position embedding
    """
    def __init__(self, in_dim, num_rqueries,num_pqueries, nheads=8):
        super(BoAQ_pe_Block, self).__init__()

        self.num_rqueries = num_rqueries
        self.num_pqueries = num_pqueries
        self.in_dim = in_dim
        self.nheads = nheads
        self.scale = in_dim ** -0.5 # sqrt(C)

        # self.proj_rq = Channel_Att(in_dim, in_dim)
        self.proj_pq = Channel_Att(in_dim, in_dim)
        
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        # self.mlp = MLP(in_dim, in_dim, in_dim)
        # self.fc_kv = nn.Linear(in_dim,in_dim *2)
        # self.rqueries = torch.nn.Parameter(torch.randn(1, num_rqueries, in_dim)) # 
        self.pqueries = torch.nn.Parameter(torch.randn(1, num_pqueries, in_dim)) # 
        
        # the following two lines are used during training only, you can cache their output in eval.
        # self.self_attn_rq = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.norm_rq = torch.nn.LayerNorm(in_dim)

        self.self_attn_pq = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_pq = torch.nn.LayerNorm(in_dim)

        #####
        # self.r_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.r_norm_out = torch.nn.LayerNorm(in_dim)
        self.p_pos_embed = PositionEmbeddingCoordsSine(1,in_dim)
        self.p_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.p_norm_out = torch.nn.LayerNorm(in_dim)
    
    def forward(self,x):
        """
        x: [B,C,R,phi]
        rot_aug: whether to roll x for rot-inv
        """
        B,C,R,Phi = x.shape
        # N = int(H*W)
        # rq = self.rqueries.repeat(B, 1, 1) # [B, nqueries=R, C]
        pq = self.pqueries.repeat(B, 1, 1) # [B, nqueries=Phi, C]
        
        # the following two lines are used during training.
        # for stability purposes 
        # rq = rq + self.self_attn_rq(rq, rq, rq)[0]
        # rq = self.norm_rq(rq)
        pq = pq + self.self_attn_pq(pq, pq, pq)[0]
        pq = self.norm_pq(pq)
        # q = self.norm_q(q).reshape(B, self.num_queries, self.nheads, C // self.nheads).permute(0,2,1,3) # [B, nheads, num_queries, C//nheads]

        # x_r = self.proj_rq(x,dim = 2) # row-mean
        # x_p = self.proj_pq(x,dim = 3) # phi-mean, [B,R,C]
        x_p = x.mean(dim = 3).permute(0,2,1).contiguous() # [B,R,C]
        # position embeding
        pe = self.p_pos_embed(torch.arange(x_p.shape[1],device = x_p.device).float().reshape(-1,1))
        x_p = x_p + pe[None,...]
        # out_r,att_r = self.r_cross_attn(rq, x_r, x_r) # out_r: [B,R,C], attr: [B,R,R] expected row-wise weight map: [B,R]
        # out_r = self.r_norm_out(out_r)
        out_p,att_p = self.p_cross_attn(pq, x_p, x_p) # out_p: [B,phi,C]
        out_p = self.p_norm_out(out_p) + x_p

        # out_rp = torch.cat([out_r, out_p],dim = 1) # [B,R+phi,C]

        

        return x,out_p,att_p # att_r: placeholder


class BoAQBlock(nn.Module):
    """
    bag of queries (Axial-enhanced version)
    """
    def __init__(self, in_dim, num_rqueries,num_pqueries, nheads=8):
        super(BoAQBlock, self).__init__()

        self.num_rqueries = num_rqueries
        self.num_pqueries = num_pqueries
        self.in_dim = in_dim
        self.nheads = nheads
        self.scale = in_dim ** -0.5 # sqrt(C)

        # self.proj_rq = Channel_Att(in_dim, in_dim)
        # self.proj_pq = Channel_Att(in_dim, in_dim)
        
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        # self.mlp = MLP(in_dim, in_dim, in_dim)
        # self.fc_kv = nn.Linear(in_dim,in_dim *2)
        # self.rqueries = torch.nn.Parameter(torch.randn(1, num_rqueries, in_dim)) # 
        self.pqueries = torch.nn.Parameter(torch.randn(1, num_pqueries, in_dim)) # 
        
        # the following two lines are used during training only, you can cache their output in eval.
        # self.self_attn_rq = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.norm_rq = torch.nn.LayerNorm(in_dim)

        self.self_attn_pq = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_pq = torch.nn.LayerNorm(in_dim)

        #####
        # self.r_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.r_norm_out = torch.nn.LayerNorm(in_dim)

        self.p_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.p_norm_out = torch.nn.LayerNorm(in_dim)
    
    def forward(self,x):
        """
        x: [B,R,C]
        rot_aug: whether to roll x for rot-inv
        """
        B,C,R,Phi = x.shape
        # B,R,C = x_p.shape
        # out_p = self.encoder(x_p)
        # N = int(H*W)
        # rq = self.rqueries.repeat(B, 1, 1) # [B, nqueries=R, C]
        pq = self.pqueries.repeat(B, 1, 1) # [B, nqueries=Phi, C]
        
        # the following two lines are used during training.
        # for stability purposes 
        # rq = rq + self.self_attn_rq(rq, rq, rq)[0]
        # rq = self.norm_rq(rq)
        pq = pq + self.self_attn_pq(pq, pq, pq)[0]
        pq = self.norm_pq(pq)
        # q = self.norm_q(q).reshape(B, self.num_queries, self.nheads, C // self.nheads).permute(0,2,1,3) # [B, nheads, num_queries, C//nheads]

        # x_r = self.proj_rq(x,dim = 2) # row-mean
        # x_p = self.proj_pq(x,dim = 3) # phi-mean, [B,R,C]
        x_p = x.mean(dim = 3).permute(0,2,1).contiguous() # [B,R,C]
        # out_r,att_r = self.r_cross_attn(rq, x_r, x_r) # out_r: [B,R,C], attr: [B,R,R] expected row-wise weight map: [B,R]
        # out_r = self.r_norm_out(out_r)
        out_p,att_p = self.p_cross_attn(pq, x_p, x_p) # out_p: [B,phi,C]
        out_p = self.p_norm_out(out_p) + x_p

        # out_rp = torch.cat([out_r, out_p],dim = 1) # [B,R+phi,C]

    
        return x,out_p,att_p # att_r: placeholder

class BoAQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_rqueries=32,num_pqueries = 32, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.BatchNorm2d(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoAQBlock(in_dim, num_rqueries,num_pqueries, nheads=in_dim // 64) for _ in range(num_layers)]) # setting nheads = 8 directly？
        
        self.fc = torch.nn.Linear(num_layers*(num_pqueries), row_dim)
        
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        # x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        # x = x.mean(dim = 3).permute(0,2,1).contiguous() # [B,R,C]
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1) # [B, blocks * (num_rqueries+num_pqueries), C]
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns

class BoAq_pe(BoAQ):
    def __init__(self, in_channels=1024, proj_channels=512, num_rqueries=32, num_pqueries=32, num_layers=2, row_dim=32):
        super().__init__(in_channels, proj_channels, num_rqueries, num_pqueries, num_layers, row_dim)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoAQ_pe_Block(in_dim, num_rqueries,num_pqueries, nheads = 8 ) for _ in range(num_layers)])
    
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        # x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        # x = x.mean(dim = 3).permute(0,2,1).contiguous() # [B,R,C]
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1) # [B, blocks * (num_rqueries+num_pqueries), C]
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns

    

if __name__ == "__main__":
    # bocq = BoCQ(in_channels = 1024, proj_channels = 512, num_queries = 32, num_layers = 1, row_dim = 4)
    boaq = BoAQ(in_channels = 1024, proj_channels = 512,num_rqueries=8, num_pqueries = 4, num_layers = 1, row_dim = 4)
    x = torch.rand((2,1024,4,8))
    out,_ = boaq(x)
    print(out.shape)

