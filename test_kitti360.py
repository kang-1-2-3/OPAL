import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import get_top_k_indices, get_top_percentage_indices,calculate_geolocalization_metrics
import yaml 
from network.network import Pcmapvprtest_boaq
from test_kitti360_loader import PcDataset_Val, collate_fn_BEV, OSMDataset_Val
from tabulate import tabulate
import os

def validate_model(model: nn.Module, pc_loader: DataLoader, osm_loader: DataLoader, device: torch.device, sequence_list: list) -> dict:
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        osm_feature_database = []
        xy_list = []
        for data in tqdm(osm_loader, desc="Processing OSM data"):
            osm_map = data['osm_map'].to(device)
            xy = data['raster_xy'].to(device)
            
            osm_desc = model(osm_map, None)
            
            osm_feature_database.append(osm_desc)
            xy_list.append(xy)
            
        osm_feature_database = torch.cat(osm_feature_database, dim=0)
        xy_list = torch.cat(xy_list, dim=0)
        
        pc_features = []
        pc_xy_list = []
        correct_xy_list = []
        incorrect_xy_list = []
        all_similarities = []

        for data in tqdm(pc_loader, desc="Processing point cloud data"):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
            train_grid_ten = [torch.from_numpy(i[:,:2]).to(device) for i in data[0]]
            train_point_label = [torch.from_numpy(i).to(device) for i in data[1]]
            pc_vis_mask = data[4].to(device)
            pc_data = (train_pt_fea_ten, train_grid_ten, train_point_label, pc_vis_mask)
            xy = data[3].to(device)
            
            pc_desc = model(None, pc_data)
            
            pc_features.append(pc_desc)
            pc_xy_list.append(xy)
            
        pc_features = torch.cat(pc_features, dim=0)
        pc_xy_list = torch.cat(pc_xy_list, dim=0)

        geo_metrics = {
            'top1': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
            'top5': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
            'top1_percent': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0}
        }
        
        for i, pc_feature in tqdm(enumerate(pc_features), desc='Calculating recall'):
            similarity_matrix = F.cosine_similarity(pc_feature.unsqueeze(0), osm_feature_database, dim=1)
            all_similarities.append(similarity_matrix.squeeze().cpu().numpy())
            
            correct_xy = pc_xy_list[i].unsqueeze(0).cpu().numpy()
            
            similarity_matrix_unsqueezed = similarity_matrix.unsqueeze(0)
            top1_indices = get_top_k_indices(similarity_matrix_unsqueezed, k=1)
            top5_indices = get_top_k_indices(similarity_matrix_unsqueezed, k=5)
            top_1_percent_indices = get_top_percentage_indices(similarity_matrix_unsqueezed, percentage=0.01)
            
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top1_indices, correct_xy, xy_list)
            geo_metrics['top1']['geo_1m_ratio'] += geo_1m
            geo_metrics['top1']['geo_5m_ratio'] += geo_5m
            geo_metrics['top1']['geo_10m_ratio'] += geo_10m
            
            if geo_5m == 1:
                correct_xy_list.append(correct_xy[0])
            else:
                incorrect_xy_list.append(correct_xy[0])
            
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top5_indices, correct_xy, xy_list)
            geo_metrics['top5']['geo_1m_ratio'] += geo_1m
            geo_metrics['top5']['geo_5m_ratio'] += geo_5m
            geo_metrics['top5']['geo_10m_ratio'] += geo_10m
            
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top_1_percent_indices, correct_xy, xy_list)
            geo_metrics['top1_percent']['geo_1m_ratio'] += geo_1m
            geo_metrics['top1_percent']['geo_5m_ratio'] += geo_5m
            geo_metrics['top1_percent']['geo_10m_ratio'] += geo_10m
            
        for key in geo_metrics:
            geo_metrics[key]['geo_1m_ratio'] /= len(pc_features)
            geo_metrics[key]['geo_5m_ratio'] /= len(pc_features)
            geo_metrics[key]['geo_10m_ratio'] /= len(pc_features)
        
    val_loss /= len(pc_loader.dataset)
    
    
    results = {
        'val_loss': val_loss,
        'geo_metrics': geo_metrics
    }
    
    headers = ["", "1m", "5m", "10m"]
    rows = [
        ["top1", 
         f"{geo_metrics['top1']['geo_1m_ratio']:.4f}", 
         f"{geo_metrics['top1']['geo_5m_ratio']:.4f}", 
         f"{geo_metrics['top1']['geo_10m_ratio']:.4f}"],
        ["top5", 
         f"{geo_metrics['top5']['geo_1m_ratio']:.4f}", 
         f"{geo_metrics['top5']['geo_5m_ratio']:.4f}", 
         f"{geo_metrics['top5']['geo_10m_ratio']:.4f}"],
        ["top1%", 
         f"{geo_metrics['top1_percent']['geo_1m_ratio']:.4f}", 
         f"{geo_metrics['top1_percent']['geo_5m_ratio']:.4f}", 
         f"{geo_metrics['top1_percent']['geo_10m_ratio']:.4f}"]
    ]
    
    print("\nGeolocation Performance Metrics:")
    print(tabulate(rows, headers, tablefmt="grid"))
    print(f"\nValidation Loss: {val_loss:.4f}")
    
    return results

if __name__ == '__main__':
    with open('conf/data/kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = config['training']['device'] 
    print('Device:', device)
    
    import argparse 
    parser = argparse.ArgumentParser(description="Process KITTI-360 sequence")
    parser.add_argument(
        "--seq", 
        type=str, 
        default="2013_05_28_drive_0000_sync"
    )
    args = parser.parse_args()
    seq = args.seq
    # eval one sequence once
    # validation_sequence_list = ["2013_05_28_drive_0000_sync","2013_05_28_drive_0005_sync","2013_05_28_drive_0006_sync","2013_05_28_drive_0009_sync"] 
    validation_sequence_list = [seq] 
    print(f"Validating on sequence: {validation_sequence_list}")

    model = Pcmapvprtest_boaq(config)
    model.load_state_dict(torch.load('checkpoints/checkpoint.pth.tar')['model_state_dict'])
    model.to(device)
    
    
    osmval_dataset = OSMDataset_Val(config, mode='val', sequence_list=validation_sequence_list)
    pcval_dataset = PcDataset_Val(config, mode='val', sequence_list=validation_sequence_list)
    
    pcval_loader = DataLoader(pcval_dataset, batch_size=config['training']['batch_size'],
                          shuffle=False, drop_last=False,
                          num_workers=config['training']['num_workers'],
                          pin_memory=True, collate_fn=collate_fn_BEV)
    osmval_loader = DataLoader(osmval_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, drop_last=False,
                            num_workers=config['training']['num_workers'],
                            pin_memory=True)
    
    print('PC Validation dataset loaded, length:', len(pcval_loader.dataset))
    print('OSM Validation dataset loaded, length:', len(osmval_loader.dataset))

    results = validate_model(model, pcval_loader, osmval_loader, device, validation_sequence_list)

