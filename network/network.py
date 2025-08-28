import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_scatter
from network.BEV_Unet import BEV_Unet_Encoder,BEV_Unet_BoAQ_pe_Encoder
from network.map_encoder import MapEncoder


class ptBEVnet(nn.Module):
    
    def __init__(self, BEV_net, grid_size, pt_model = 'pointnet', fea_dim = 3, pt_pooling = 'max', kernal_size = 3,
                 out_pt_fea_dim = 64, max_pt_per_encode = 64, cluster_num = 4, pt_selection = 'farthest', fea_compre = None):
        super(ptBEVnet, self).__init__()
        assert pt_pooling in ['max']
        assert pt_selection in ['random','farthest']
        
        if pt_model == 'pointnet':

            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(fea_dim),
                
                nn.Linear(fea_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                
                nn.Linear(64, out_pt_fea_dim)
            )
        self.pt_model = pt_model
        self.BEV_model = BEV_net
        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.pt_selection = pt_selection
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.semantic_embedding = nn.Embedding(32, 16)
        # NN stuff
        if kernal_size != 1:
            if self.pt_pooling == 'max':
                self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size-1)//2, dilation=1)
            else: raise NotImplementedError
        else: self.local_pool_op = None
        
        # parametric pooling        
        if self.pt_pooling == 'max':
            self.pool_dim = out_pt_fea_dim
        
        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, pt_label, voxel_fea=None, intensity=None,return_lf = False,agg = None,dim = None):
        cur_dev = pt_fea[0].device
        
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch)) # add batch info

        cat_pt_fea = torch.cat(pt_fea,dim = 0)
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)
        cat_pt_label = torch.cat(pt_label, dim=0)
        pt_num = cat_pt_ind.shape[0] # 249273

        shuffled_ind = torch.randperm(pt_num)
        cat_pt_fea = cat_pt_fea[shuffled_ind,:]
        cat_pt_ind = cat_pt_ind[shuffled_ind,:]
        cat_pt_label = cat_pt_label[shuffled_ind,:]

        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0) 
        unq = unq.type(torch.int64)
        
        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
            
        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        cat_pt_label = cat_pt_label[remain_ind,:]
        # cat_pt_intensity = cat_pt_intensity[remain_ind].unsqueeze(1)
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)

        cat_pt_label = torch.squeeze(self.semantic_embedding(cat_pt_label))# [249273,16]
        # cat_pt_fea = torch.cat([cat_pt_fea, cat_pt_intensity, cat_pt_label], dim=-1)
        cat_pt_fea = torch.cat([cat_pt_fea, cat_pt_label], dim=-1)
        # process feature
        if self.pt_model == 'pointnet':
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea) # [249273, 512]
        
        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # [49953, 512]
        else: raise NotImplementedError
        
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)  # [49953, 32]
        else:
            processed_pooled_data = pooled_data # [49953, 32]
        
        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim] # [2, 480, 360, 32]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data # [2, 480, 360, 32]
        out_data = out_data.permute(0,3,1,2)
        if self.local_pool_op != None:
            out_data = self.local_pool_op(out_data) # [2, 8, 480, 360]
        
        if (voxel_fea is not None):
            voxel_fea = voxel_fea.unsqueeze(1) # [2, 1, 480, 360]
            out_data = torch.cat((out_data, voxel_fea), 1) # [2, 9, 480, 360]

        out_data = self.BEV_model(out_data,return_lf = return_lf,agg = agg,dim = dim) # [2, 512]

        return out_data
    
def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)



class OSM_Encoder(nn.Module):
    def __init__(self, conf, BEV_net=None):
        super(OSM_Encoder, self).__init__()
        self.map_encoder = MapEncoder(conf['model']['map_encoder'])
        self.radial_resolution = 480
        self.angular_resolution = 360
        
        self.BEV_model = BEV_net

    def cartesian_to_polar(self, map_data, building_mask=None,roll = 0):
        B, C, H, W = map_data.shape
        center_x, center_y = W // 2, H // 2
        theta = torch.linspace(-np.pi, np.pi, self.angular_resolution, device=map_data.device)
        radius = torch.linspace(0, center_x, self.radial_resolution, device=map_data.device)

        grid_x = center_x + radius.view(-1, 1) * torch.cos(theta).view(1, -1)
        grid_y = center_y - radius.view(-1, 1) * torch.sin(theta).view(1, -1)  
        
        grid_x = grid_x.view(1, self.radial_resolution, self.angular_resolution, 1).expand(B, -1, -1, -1)
        grid_x = grid_x / (W - 1) * 2 - 1 
        
        grid_y = grid_y.view(1, self.radial_resolution, self.angular_resolution, 1).expand(B, -1, -1, -1)
        grid_y = grid_y / (H - 1) * 2 - 1 
        grid = torch.cat((grid_x, grid_y), dim=-1)
        
        polar_map_data = F.grid_sample(map_data, grid, mode='bilinear', align_corners=True)  # [B, C, R, A]
        
        if roll > 0:
            polar_map_data = torch.roll(polar_map_data,shifts=roll.item(),dims = -1)

        return polar_map_data

    def generate_visibility_mask(self, map_data,roll = 0):
        building_mask = (map_data[:, 1] == 5).float().unsqueeze(1)  # [B,1,H,W]
        polar_building_mask = self.cartesian_to_polar(building_mask,roll = roll)  # [B,1,R,A]
        polar_building_mask = polar_building_mask.squeeze(1)  # [B,R,A]

        threshold = 0.5
        building_bool = (polar_building_mask > threshold).float() 
        cumsum_mask = torch.cumsum(building_bool, dim=1) 
        first_building = (cumsum_mask == 1) & (building_bool == 1) 

        first_building_dist = torch.argmax(first_building.float(), dim=1)  # [B,A]
        has_building = torch.any(building_bool, dim=1)  # [B,A]
        
        first_building_dist = torch.where(has_building, 
                                        first_building_dist, 
                                        torch.full_like(first_building_dist, self.radial_resolution - 1))
        
        radial_indices = torch.arange(self.radial_resolution, device=map_data.device)
        radial_indices = radial_indices.view(1, -1, 1).expand(-1, -1, self.angular_resolution)
        
        first_building_dist = first_building_dist.unsqueeze(1)  # [B,1,A]
        visibility_mask = (radial_indices < first_building_dist).float()
        
        return visibility_mask.unsqueeze(1)

    def forward(self, map_data,roll = 0,return_lf = False,agg = None,dim = None):
        building_mask = self.generate_visibility_mask(map_data,roll = roll)
        map_data = self.map_encoder(map_data) # [B, 48, 200, 200]
        # map_data = self.cartesian_to_polar(map_data, building_mask) # [B, 9, 480, 360]
        map_data = self.cartesian_to_polar(map_data,roll = roll)
        # map_data = self.map_encoder(map_data)
        # visibility_head = self.visibility_predictor(map_data)
        map_data = torch.cat((map_data, building_mask), dim=1)  # [B, 49, 480, 360]
        map_global_descriptor = self.BEV_model(map_data,return_lf = return_lf, agg = agg, dim = dim) # [B, 512]
        return map_global_descriptor
    
    def forward_gap(self, map_data, roll = 0):
         building_mask = self.generate_visibility_mask(map_data,roll = roll)
         map_data = self.map_encoder(map_data) # [B, 48, 200, 200]
         # map_data = self.cartesian_to_polar(map_data, building_mask) # [B, 9, 480, 360]
         map_data = self.cartesian_to_polar(map_data,roll=roll)
         # map_data = self.map_encoder(map_data)
         # visibility_head = self.visibility_predictor(map_data)
         map_data = torch.cat((map_data, building_mask), dim=1)  # [B, 49, 480, 360]
         map_global_descriptor = self.BEV_model.forward_gap(map_data) # [B, 512]
         return map_global_descriptor
    
    def forward_am(self, map_data, roll = 0,dim = -1):
         building_mask = self.generate_visibility_mask(map_data,roll = roll)
         map_data = self.map_encoder(map_data) # [B, 48, 200, 200]
         # map_data = self.cartesian_to_polar(map_data, building_mask) # [B, 9, 480, 360]
         map_data = self.cartesian_to_polar(map_data,roll=roll)
         map_data = torch.cat((map_data, building_mask), dim=1)  # [B, 49, 480, 360]
         map_global_descriptor = self.BEV_model.forward_am(map_data,dim) # [B, 512]
         return map_global_descriptor
    
    def forward_womask(self, map_data):
        map_data = self.map_encoder(map_data) # embedding
        map_data = self.cartesian_to_polar(map_data) # transfer to polar coordinate system
        map_global_descriptor = self.BEV_model(map_data) # [B, 512]
        return map_global_descriptor
    
    def forward_roll(self, map_data,rolls = [0]):
        # rolls = [0,torch.pi * 0.5,torch.pi,torch.pi * 1.5]
        map_global_descriptor_rots = []
        for r in rolls:
            descriptor = self.forward(map_data,r)
            map_global_descriptor_rots.append(descriptor)
        return map_global_descriptor_rots




    


class Pcmapvpr_boaq(nn.Module):
    def __init__(self, conf):
        super(Pcmapvpr_boaq, self).__init__()
        self.pc_encoder = ptBEVnet(BEV_net=BEV_Unet_BoAQ_pe_Encoder(n_height = 33, input_batch_norm = True, circular_padding = True), grid_size=[480,360,32], pt_model='pointnet', fea_dim=24, pt_pooling='max', kernal_size=1, out_pt_fea_dim=64, max_pt_per_encode=256, cluster_num=4, pt_selection='random', fea_compre=32)
        self.map_encoder = OSM_Encoder(conf, BEV_net=BEV_Unet_BoAQ_pe_Encoder(n_height = 49, input_batch_norm = True, circular_padding = True))
        
    def forward(self, osm_data, pc_data):
        osm_desc = self.map_encoder(osm_data) # [B, 1, 480, 360]
        # remove pc voxel feature
        pc_desc = self.pc_encoder(pc_data[0], pc_data[1], pc_data[2], pc_data[3]) # [B, 512]
        osm_desc = F.normalize(osm_desc, p=2, dim=1)
        pc_desc = F.normalize(pc_desc, p=2, dim=1)
        return osm_desc, pc_desc
    
    def forward_vis(self, osm_data, pc_data):
        osm_desc,osm_feat = self.map_encoder(osm_data,return_lf = True) # [B, 1, 480, 360]
        # remove pc voxel feature
        pc_desc,pc_feat = self.pc_encoder(pc_data[0], pc_data[1], pc_data[2], pc_data[3],return_lf = True) # [B, 512]
        osm_desc = F.normalize(osm_desc, p=2, dim=1)
        pc_desc = F.normalize(pc_desc, p=2, dim=1)
        return osm_desc,osm_feat, pc_desc, pc_feat

class Pcmapvprtest(nn.Module):
    def __init__(self, conf):
        super(Pcmapvprtest, self).__init__()
        self.pc_encoder = ptBEVnet(BEV_net=BEV_Unet_Encoder(n_height = 33, input_batch_norm = True, circular_padding = True), grid_size=[480,360,32], pt_model='pointnet', fea_dim=24, pt_pooling='max', kernal_size=1, out_pt_fea_dim=64, max_pt_per_encode=256, cluster_num=4, pt_selection='random', fea_compre=32)
        self.map_encoder = OSM_Encoder(conf, BEV_net=BEV_Unet_Encoder(n_height = 49, input_batch_norm = True, circular_padding = True))
        
    def forward(self, osm_data, pc_data):
        if osm_data is not None:
            osm_desc = self.map_encoder(osm_data)
            osm_desc = F.normalize(osm_desc, p=2, dim=1)
            return osm_desc
        if pc_data is not None:
            pc_desc = self.pc_encoder(pc_data[0], pc_data[1], pc_data[2], pc_data[3])
            pc_desc = F.normalize(pc_desc, p=2, dim=1)
            return pc_desc
        
class Pcmapvprtest_boaq(Pcmapvprtest):
    def __init__(self, conf):
        super(Pcmapvprtest_boaq, self).__init__(conf)
        self.pc_encoder = ptBEVnet(BEV_net=BEV_Unet_BoAQ_pe_Encoder(n_height = 33, input_batch_norm = True, circular_padding = True), grid_size=[480,360,32], pt_model='pointnet', fea_dim=24, pt_pooling='max', kernal_size=1, out_pt_fea_dim=64, max_pt_per_encode=256, cluster_num=4, pt_selection='random', fea_compre=32)
        self.map_encoder = OSM_Encoder(conf, BEV_net=BEV_Unet_BoAQ_pe_Encoder(n_height = 49, input_batch_norm = True, circular_padding = True))
    
    