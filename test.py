import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from network.network import Pcmapvpr
from kitti_dataloader import PcMapLocDataset
from tqdm import tqdm
import torch.nn.functional as F
import os
from datetime import datetime
from train import get_top_k_indices, get_top_percentage_indices, get_top_k_ratio, set_seed


def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        osm_feture_database = []
        pc_features = []   
        for data in tqdm(test_loader, desc="Testing"):
            osm_data, pc_bev_data = data['osm_map'].to(device), data['pc_bev_img'].to(device) 
            osm_descs, pc_bev_descs = model(osm_data, pc_bev_data)
            pad_size = 8192 - 1024
            osm_descs = F.pad(osm_descs, (0, pad_size), mode='constant', value=0)
            osm_feture_database.append(osm_descs)
            pc_features.append(pc_bev_descs)
        osm_feture_database = torch.cat(osm_feture_database, dim=0)
        pc_features = torch.cat(pc_features, dim=0)
        
        # Initialize metrics
        top1_ratio = 0.0
        top5_ratio = 0.0
        top_1_percent_ratio = 0.0
        
        # Loop through each pc_feature
        for pc_feature in pc_features:
            # Calculate cosine similarity
            similarity_matrix = F.cosine_similarity(pc_feature.unsqueeze(0), osm_feture_database.unsqueeze(0), dim=2)
            
            # Get top k and top percentage k indices
            top1_indices = get_top_k_indices(similarity_matrix, k=1)
            top5_indices = get_top_k_indices(similarity_matrix, k=5)
            top_1_percent_indices = get_top_percentage_indices(similarity_matrix, percentage=0.01)
            
            # Calculate ratios
            top1_ratio += get_top_k_ratio(similarity_matrix, top1_indices)
            top5_ratio += get_top_k_ratio(similarity_matrix, top5_indices)
            top_1_percent_ratio += get_top_k_ratio(similarity_matrix, top_1_percent_indices)
        
        # Average the ratios
        top1_ratio /= len(pc_features)
        top5_ratio /= len(pc_features)
        top_1_percent_ratio /= len(pc_features)
       
    results = {
        'top1_ratio': top1_ratio,
        'top5_ratio': top5_ratio,
        'top_1_percent_ratio': top_1_percent_ratio
    }
    
    return results

if __name__ == "__main__":
    # load config
    with open('conf/data/kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # set seed
    set_seed(seed = config['training']['seed'])

    print('Device:', config['training']['device'])
    # model, criterion
    model = Pcmapvpr(config)
    
    # data loader
    test_dataset = PcMapLocDataset(config, 'test')  
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True, num_workers=config['training']['num_workers'], pin_memory=True)
    print('Test dataset loaded, length:', len(test_loader.dataset))

    # device
    device = torch.device(config['training']['device'])
    model.to(device)
    
    # load best model checkpoint
    checkpoint = torch.load('checkpoints/20241206_195720/checkpoint_epoch_2.pth.tar')
    print('top1 ratio', checkpoint['top_1_ratio'])
    print('top5 ratio', checkpoint['top_5_ratio'])
    print('top1 percent ratio', checkpoint['top_1_percent_ratio'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # testing process
    test_results = test_model(model, test_loader, device)
    print(f"Top 1 Ratio: {test_results['top1_ratio']:.4f}")
    print(f"Top 5 Ratio: {test_results['top5_ratio']:.4f}")
    print(f"Top 1% Ratio: {test_results['top_1_percent_ratio']:.4f}")
