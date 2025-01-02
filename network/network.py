import sys
sys.path.append('/data/Pcmaploc/code/geo-localization-with-point-clouds-and-openstreetmap')
from maploc.models.map_encoder_modified import MapEncoder   
import torch 
import torch.nn as nn
import torch.nn.functional as F
from BEVPlace.REIN import REIN, NetVLAD
import yaml
import numpy as np
from .pointpillar import PointPillarEncoder
# TODO: Implement the network class, including OSM Encoder (Modify Orienter Net), Point Cloud Encoder (BEVPlace++) 

class Pcmapvpr(nn.Module):
    def __init__(self, conf):
        super(Pcmapvpr, self).__init__()
        
        self.map_encoder = nn.Sequential(
            MapEncoder(conf['model']['map_encoder']),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        
        # Add NetVLAD layer for map features
        self.map_vlad = NetVLAD(num_clusters=32, dim=8)

        # Fully connected layers for map features
        self.map_fc = nn.Sequential(
            nn.Linear(8 * 32 * 32, 4096),  
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Initialize the Point Cloud Encoder (BEVPlace++)
        self.pc_encoder = REIN()

        self.pc_conv1 = nn.Sequential(
            nn.Conv1d(8192, 4096, kernel_size=1),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.pc_conv2 = nn.Sequential(
            nn.Conv1d(4096, 2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.pc_conv3 = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.pc_conv4 = nn.Conv1d(1024, 256, kernel_size=1)
        
    def forward(self, map_data, pc_data):
        # Encode the map data with combined operations
        map_features = self.map_encoder[0](map_data)
        map_features_pooled = self.map_encoder[1:](map_features['map_features']['feature_maps'][0])
        
        # Apply NetVLAD directly to normalized features
        map_features_vlad = self.map_vlad(map_features_pooled)
        map_features_vlad = F.normalize(map_features_vlad, p=2, dim=1)  # L2 normalize VLAD features

        # Encode the point cloud data
        _, _, global_pc_descs = self.pc_encoder(pc_data)
        global_pc_descs_flat = global_pc_descs.view(global_pc_descs.size(0), -1)  # Flatten

        global_pc_descs_flat = global_pc_descs_flat.unsqueeze(2)  # Add channel dimension
        global_pc_descs_flat = self.pc_conv1(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv2(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv3(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv4(global_pc_descs_flat)
        global_pc_descs_flat = global_pc_descs_flat.squeeze(2)  # Remove channel dimension

        global_pc_descs_flat = F.normalize(global_pc_descs_flat, p=2, dim=1)  # Normalize
        
        # print(map_features_fc.shape, global_pc_descs_flat.shape) # (1, 1024), (1, 8192)
        # return map_features_fc, global_pc_descs_flat
        return map_features_vlad, global_pc_descs_flat


# for pointpillar
class PcmapvprV2(nn.Module):
    def __init__(self, conf):
        super(PcmapvprV2, self).__init__()
        
        self.map_encoder = nn.Sequential(
            MapEncoder(conf['model']['map_encoder']),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        
        # Add NetVLAD layer for map features
        self.map_vlad = NetVLAD(num_clusters=32, dim=8)

        # # Fully connected layers for map features
        # self.map_fc = nn.Sequential(
        #     nn.Linear(8 * 32 * 32, 4096),  
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024), 
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )

        # Initialize the Point Cloud Encoder (PointPillar)
        self.pc_encoder = PointPillarEncoder(128, [-30.0, 30.0, 0.15], [-15.0, 15.0, 0.15], [-10.0, 10.0, 20.0])
        self.pc_vlad = NetVLAD(num_clusters=64, dim=128)
        self.pc_conv1 = nn.Sequential(
            nn.Conv1d(8192, 4096, kernel_size=1),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.pc_conv2 = nn.Sequential(
            nn.Conv1d(4096, 2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.pc_conv3 = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.pc_conv4 = nn.Conv1d(1024, 256, kernel_size=1)
        
    def forward(self, map_data, pc_data, pc_mask):
        # Encode the map data with combined operations
        map_features = self.map_encoder[0](map_data)
        map_features_pooled = self.map_encoder[1:](map_features['map_features']['feature_maps'][0])
        
        # Apply NetVLAD directly to normalized features
        map_features_vlad = self.map_vlad(map_features_pooled)
        map_features_vlad = F.normalize(map_features_vlad, p=2, dim=1)  # L2 normalize VLAD features [batch, 256]

        # Encode the point cloud data
        pc_features = self.pc_encoder(pc_data, pc_mask) # [batch, 128, 200, 400]
        global_pc_descs = self.pc_vlad(pc_features).unsqueeze(2) # [batch, 8192]
        global_pc_descs_flat = self.pc_conv1(global_pc_descs)
        global_pc_descs_flat = self.pc_conv2(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv3(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv4(global_pc_descs_flat)
        global_pc_descs_flat = global_pc_descs_flat.squeeze(2)  

        global_pc_descs_flat = F.normalize(global_pc_descs_flat, p=2, dim=1)
        return map_features_vlad, global_pc_descs_flat

# for triplet loss
class PcmapvprV3(nn.Module):
    def __init__(self, conf):
        super(PcmapvprV3, self).__init__()
        
        self.map_encoder = nn.Sequential(
            MapEncoder(conf['model']['map_encoder']),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        
        # Add NetVLAD layer for map features
        self.map_vlad = NetVLAD(num_clusters=32, dim=8)

        # Initialize the Point Cloud Encoder (BEVPlace++)
        self.pc_encoder = REIN()

        self.pc_conv1 = nn.Sequential(
            nn.Conv1d(8192, 4096, kernel_size=1),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.pc_conv2 = nn.Sequential(
            nn.Conv1d(4096, 2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.pc_conv3 = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.pc_conv4 = nn.Conv1d(1024, 256, kernel_size=1)
        
    def forward(self, pos_map_data, neg_map_data, pc_data):
        # Encode the positive map data
        pos_map_features = self.map_encoder[0](pos_map_data)
        pos_map_features_pooled = self.map_encoder[1:](pos_map_features['map_features']['feature_maps'][0])
        pos_map_features_vlad = self.map_vlad(pos_map_features_pooled)
        pos_map_features_vlad = F.normalize(pos_map_features_vlad, p=2, dim=1)  # L2 normalize VLAD features

        # Encode the negative map data
        neg_map_features = self.map_encoder[0](neg_map_data)
        neg_map_features_pooled = self.map_encoder[1:](neg_map_features['map_features']['feature_maps'][0])
        neg_map_features_vlad = self.map_vlad(neg_map_features_pooled)
        neg_map_features_vlad = F.normalize(neg_map_features_vlad, p=2, dim=1)  # L2 normalize VLAD features

        # Encode the point cloud data
        _, _, global_pc_descs = self.pc_encoder(pc_data)
        global_pc_descs_flat = global_pc_descs.view(global_pc_descs.size(0), -1)  # Flatten

        global_pc_descs_flat = global_pc_descs_flat.unsqueeze(2)  # Add channel dimension
        global_pc_descs_flat = self.pc_conv1(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv2(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv3(global_pc_descs_flat)
        global_pc_descs_flat = self.pc_conv4(global_pc_descs_flat)
        global_pc_descs_flat = global_pc_descs_flat.squeeze(2) 
        global_pc_descs_flat = F.normalize(global_pc_descs_flat, p=2, dim=1)  
        
        return pos_map_features_vlad, neg_map_features_vlad, global_pc_descs_flat

# for dim=8192
class PcmapvprV4(nn.Module):
    def __init__(self, conf):
        super(Pcmapvpr, self).__init__()
        
        self.map_encoder = nn.Sequential(
            MapEncoder(conf['model']['map_encoder']),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        
        # Add NetVLAD layer for map features
        self.map_vlad = NetVLAD(num_clusters=32, dim=8)

        # Fully connected layers for map features (升维到 8192)
        self.map_fc = nn.Sequential(
            nn.Linear(8 * 32 * 32, 4096),  
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),  # 直接升维到 8192
            nn.BatchNorm1d(8192),
            nn.ReLU()
        )

        # Initialize the Point Cloud Encoder (BEVPlace++)
        self.pc_encoder = REIN()

        
    def forward(self, map_data, pc_data):
        # Encode the map data
        map_features = self.map_encoder[0](map_data)
        map_features_pooled = self.map_encoder[1:](map_features['map_features']['feature_maps'][0])
        
        # Apply NetVLAD to normalized features
        map_features_vlad = self.map_vlad(map_features_pooled)
        map_features_vlad = F.normalize(map_features_vlad, p=2, dim=1)  # L2 normalize VLAD features
        
        # Pass through map_fc and project to 8192 dimensions
        map_features_vlad = self.map_fc(map_features_vlad)
        map_features_vlad = F.normalize(map_features_vlad, p=2, dim=1)  # Normalize to unit length

        # Encode the point cloud data
        _, _, global_pc_descs = self.pc_encoder(pc_data)
 

        return map_features_vlad, global_pc_descs
    
# Example usage
if __name__ == "__main__":
    # Load configuration
    with open('conf/data/kitti.yaml', 'r') as file:
        conf = yaml.safe_load(file)
    # Initialize the model
    device = torch.device(conf['training']['device'])
    model = PcmapvprV2(conf).to(device)
    
    mode = 'val'
    from kitti_dataloader import PcMapLocDataset
    dataset = PcMapLocDataset(conf, mode=mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    first_batch = next(iter(dataloader))
    lidar_data = first_batch['lidar_data'].to(device)
    lidar_mask = first_batch['lidar_mask'].to(device)
    osm_map = first_batch['osm_map'].to(device)
    output = model(osm_map, lidar_data, lidar_mask)
    print(1)