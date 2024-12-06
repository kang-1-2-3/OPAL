import sys
sys.path.append('D:\\GitHub\\geo-localization-with-point-clouds-and-openstreetmap')
from maploc.models.map_encoder_modified import MapEncoder   
import torch 
import torch.nn as nn
from BEVPlace.REIN import REIN
import yaml
import numpy as np
# TODO: Implement the network class, including OSM Encoder (Modify Orienter Net), Point Cloud Encoder (BEVPlace++) 

class Pcmapvpr(nn.Module):
    def __init__(self, conf):
        super(Pcmapvpr, self).__init__()
        
        # Initialize the OSM Encoder (Map Encoder) and pooling layers
        self.map_encoder = nn.Sequential(
            MapEncoder(conf['model']['map_encoder']),
            nn.MaxPool2d(kernel_size=4, stride=4)  
        )

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
        
    def forward(self, map_data, pc_data):
        # Encode the map data
        map_features = self.map_encoder[0](map_data)  # Pass through MapEncoder
        map_features_pooled = self.map_encoder[1](map_features['map_features']['feature_maps'][0])
        map_features_flat = map_features_pooled.view(map_features_pooled.size(0), -1)
        map_features_fc = self.map_fc(map_features_flat)  # Fully connected layers

        # Encode the point cloud data
        _, _, global_pc_descs = self.pc_encoder(pc_data)
        global_pc_descs_flat = global_pc_descs.view(global_pc_descs.size(0), -1)  # Flatten
        
        # print(map_features_fc.shape, global_pc_descs_flat.shape) # (1, 1024), (1, 8192)
        return map_features_fc, global_pc_descs_flat


# Example usage
if __name__ == "__main__":
    # Load configuration
    with open('conf/data/kitti.yaml', 'r') as file:
        conf = yaml.safe_load(file)
    # Initialize the model
    model = Pcmapvpr(conf)
    
    # Example map data and point cloud data
    # example_map_data = torch.randn(1, 3, 128, 128).long()
    example_map_data = torch.from_numpy(np.ascontiguousarray(np.load('canvas_00_0.npy'))).long().unsqueeze(0)
    example_map_data = example_map_data.repeat(4, 1, 1, 1)  # Repeat to create batch size of 4

    import cv2
    example_pc_data = cv2.imread('pc_bev_img_00_0.png')
    example_pc_data = example_pc_data.transpose(2, 0, 1)
    example_pc_data = torch.from_numpy((example_pc_data.astype(np.float32))/256).unsqueeze(0)
    example_pc_data = example_pc_data.repeat(4, 1, 1, 1)  # Repeat to create batch size of 4
    # print(example_map_data.shape, example_pc_data.shape)
    # Forward pass
    output = model(example_map_data, example_pc_data)
    print(output)