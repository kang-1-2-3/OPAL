import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from loss import CircleLoss
from network.network import Pcmapvpr_boaq
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn.functional as F
import random
import numpy as np
from kitti_dataloader import PcMapLocDataset, collate_fn_BEV
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import argparse

def get_top_k_indices(similarity_matrix, k):
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    return top_k_indices


def get_top_percentage_indices(similarity_matrix, percentage=0.01):
    k = max(1, int(similarity_matrix.size(1) * percentage))
    return get_top_k_indices(similarity_matrix, k)

def get_top_k_ratio(top_k_indices, correct_index):
    return (top_k_indices == correct_index).any().item()

def calculate_geolocalization_metrics(top_indices, correct_coords, xy_list,return_dis = False):
    geo_1m = 0
    geo_5m = 0
    geo_10m = 0
    for idx in top_indices[0]:
        top_coords = xy_list[idx].cpu().numpy()
        distance = np.linalg.norm(top_coords - correct_coords)
        if distance <= 1:
            geo_1m = 1
        if distance <= 5:
            geo_5m = 1
        if distance <= 10:
            geo_10m = 1
    if return_dis:
        return geo_1m, geo_5m, geo_10m, distance
    else:
        return geo_1m, geo_5m, geo_10m

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    val_loss = 0.0
    predictions = []
    with torch.no_grad():
        osm_feture_database = []
        pc_features = []
        xy_list = []   
        for step, data in enumerate(tqdm(val_loader, desc="Validation", ncols=100)):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
            train_grid_ten = [torch.from_numpy(i[:,:2]).to(device) for i in data[0]]
            train_point_label = [torch.from_numpy(i).to(device) for i in data[1]]
            pc_vis_mask = data[5].to(device)
            # intensity = [torch.from_numpy(i).to(device) for i in data[6]]
            pc_data = (train_pt_fea_ten, train_grid_ten, train_point_label, pc_vis_mask)
            xy, osm_map = data[4].to(device), data[3].to(device)

            osm_desc, pc_desc = model(osm_map, pc_data)
            
            osm_feture_database.append(osm_desc)
            pc_features.append(pc_desc)
            xy_list.append(xy)
            loss = criterion(osm_desc, pc_desc)
            val_loss += loss.item() * osm_map.size(0)
            
            # if writer:
            #     writer.add_scalar(f'Loss/train_epoch_{epoch}', loss, step)
        
        osm_feture_database = torch.cat(osm_feture_database, dim=0)
        pc_features = torch.cat(pc_features, dim=0)
        xy_list = torch.cat(xy_list, dim=0)

        # Initialize metrics
        top_1_ratio = 0.0
        top_5_ratio = 0.0
        top_1_percent_ratio = 0.0
        geo_metrics = {
            'top1': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
            'top5': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
            'top1_percent': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0}
        }
        
        # Loop through each pc_feature
        for i, pc_feature in enumerate(pc_features):
            # Calculate cosine similarity
            similarity_matrix = F.cosine_similarity(pc_feature.unsqueeze(0), osm_feture_database.unsqueeze(0), dim=2)
            correct_index = torch.tensor([i], device=similarity_matrix.device)
            
            # Get top k and top percentage k indices
            top1_indices = get_top_k_indices(similarity_matrix, k=1)
            top5_indices = get_top_k_indices(similarity_matrix, k=5)
            top_1_percent_indices = get_top_percentage_indices(similarity_matrix, percentage=0.01)
            
            # Calculate ratios
            top_1_ratio += get_top_k_ratio(top1_indices, correct_index)
            top_5_ratio += get_top_k_ratio(top5_indices, correct_index)
            top_1_percent_ratio += get_top_k_ratio(top_1_percent_indices, correct_index)
            
            # Calculate geolocalization distances
            correct_coords = xy_list[correct_index].cpu().numpy()
            
            # Check top 1
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top1_indices, correct_coords, xy_list)
            geo_metrics['top1']['geo_1m_ratio'] += geo_1m
            geo_metrics['top1']['geo_5m_ratio'] += geo_5m
            geo_metrics['top1']['geo_10m_ratio'] += geo_10m
            
            # Save predictions if required
            # if return_predictions and geo_5m == 1:
            #     predictions.append((i, osm_data_list[i].cpu(), pc_bev_img_list[i].cpu(), np.linalg.norm(correct_coords-xy_list[top1_indices[0].item()].cpu().numpy())))
            # Check top 5
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top5_indices, correct_coords, xy_list)
            geo_metrics['top5']['geo_1m_ratio'] += geo_1m
            geo_metrics['top5']['geo_5m_ratio'] += geo_5m
            geo_metrics['top5']['geo_10m_ratio'] += geo_10m
            
            # Check top 1 percent
            geo_1m, geo_5m, geo_10m = calculate_geolocalization_metrics(top_1_percent_indices, correct_coords, xy_list)
            geo_metrics['top1_percent']['geo_1m_ratio'] += geo_1m
            geo_metrics['top1_percent']['geo_5m_ratio'] += geo_5m
            geo_metrics['top1_percent']['geo_10m_ratio'] += geo_10m
            
            
        
        # Average the ratios
        top_1_ratio /= len(pc_features)
        top_5_ratio /= len(pc_features)
        top_1_percent_ratio /= len(pc_features)
        for key in geo_metrics:
            geo_metrics[key]['geo_1m_ratio'] /= len(pc_features)
            geo_metrics[key]['geo_5m_ratio'] /= len(pc_features)
            geo_metrics[key]['geo_10m_ratio'] /= len(pc_features)
        
    val_loss /= len(val_loader.dataset)
    
    results = {
        'val_loss': val_loss,
        'top_1_ratio': top_1_ratio,
        'top_5_ratio': top_5_ratio,
        'top_1_percent_ratio': top_1_percent_ratio,
        'geo_metrics': geo_metrics
    }
    
    print(f"Validation Loss: {val_loss:.4f}, geo_5m: {geo_metrics['top1']['geo_5m_ratio']:.4f},  {geo_metrics['top5']['geo_5m_ratio']:.4f},  {geo_metrics['top1_percent']['geo_5m_ratio']:.4f}")
    
    return results

def save_predictions(predictions, epoch):
    from maploc.osm.viz import Colormap
    import shutil

    # Create a directory for the current epoch
    epoch_dir = f"predictions/epoch_{epoch}"
    if os.path.exists(epoch_dir):
        shutil.rmtree(epoch_dir)
    os.makedirs(epoch_dir, exist_ok=True)

    for i, osm_data, pc_bev_img, dist in predictions:
        osm_img_path = f"{epoch_dir}/{i}_osm_map_dist_{dist:.2f}.png"
        pc_bev_img_path = f"{epoch_dir}/{i}_pc_bev_img_dist_{dist:.2f}.png"
        
        # Save OSM data
        osm_data = Colormap.apply(osm_data.numpy())
        osm_img = osm_data
        plt.imsave(osm_img_path, osm_img)
        
        # Save PC BEV image
        pc_bev_img = pc_bev_img.numpy().transpose(1, 2, 0)/255
        plt.imsave(pc_bev_img_path, pc_bev_img)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, scheduler:optim.lr_scheduler._LRScheduler,config: dict):
    num_epochs = config['training']['num_epochs']
    device = torch.device(config['training']['device'])
    checkpoint_dir = config['training']['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_subdir = os.path.join(checkpoint_dir, current_time)
    os.makedirs(checkpoint_subdir)
    
    log_dir = os.path.join("runs", current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_top1_ratio = 0.0  
    best_top5_ratio = 0.0  
    best_top_1_percent_ratio = 0.0  
    best_epoch = 0  
    best_dict = {}

    best_geo_metrics = {
        'top1': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
        'top5': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
        'top1_percent': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0}
    }
    model.to(device)
    
    writer = SummaryWriter(log_dir, filename_suffix=current_time)
    
    validate_model(model, val_loader, criterion, device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs - 1}", ncols=100)):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
            train_grid_ten = [torch.from_numpy(i[:,:2]).to(device) for i in data[0]]
            train_point_label = [torch.from_numpy(i).to(device) for i in data[1]]
            pc_vis_mask = data[5].to(device)
            pc_data = (train_pt_fea_ten, train_grid_ten, train_point_label, pc_vis_mask)
            xy, osm_map = data[4].to(device), data[3].to(device)

            optimizer.zero_grad()
            osm_desc, pc_desc = model(osm_map, pc_data)
            loss = criterion(osm_desc, pc_desc)
            loss.backward()

            optimizer.step()
            running_loss += loss.item() * osm_map.size(0)

        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]}')
        writer.add_scalar('Training Loss', epoch_loss, epoch)

        if epoch % 1 == 0:
            val_results = validate_model(model, val_loader, criterion,device)
            val_loss = val_results['val_loss']
            
            writer.add_scalar('Validation Loss', val_loss, epoch)
            
            for k in ['top1', 'top5', 'top1_percent']:
                writer.add_scalar(f'Geo_Recall/{k}/1m', val_results['geo_metrics'][k]['geo_1m_ratio'], epoch)
                writer.add_scalar(f'Geo_Recall/{k}/5m', val_results['geo_metrics'][k]['geo_5m_ratio'], epoch)
                writer.add_scalar(f'Geo_Recall/{k}/10m', val_results['geo_metrics'][k]['geo_10m_ratio'], epoch)
            
            writer.add_scalar('Recall/top1', val_results['top_1_ratio'], epoch)
            writer.add_scalar('Recall/top5', val_results['top_5_ratio'], epoch)
            writer.add_scalar('Recall/top1_percent', val_results['top_1_percent_ratio'], epoch)
            print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')
        
        save_checkpoint({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, filename=f'{checkpoint_subdir}/checkpoint_epoch_{epoch}.pth.tar')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, filename=f'{checkpoint_subdir}/best_checkpoint.pth.tar')
        
    writer.close()
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001, help="Learning rate for training")
    parser.add_argument('--dataset', type=str, required=False, default='KITTI', help="Dataset to use: KITTI or KITTI360") # use KITTI for training
    args = parser.parse_args()

    # load config
    with open('conf/data/kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # set seed
    set_seed(seed = config['training']['seed'])

    print('Device:', config['training']['device'])
    # model, criterion, optimizer
    model = Pcmapvpr_boaq(config)
    criterion = CircleLoss()
    config['training']['lr'] = args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # data loader  
    train_dataset = PcMapLocDataset(config, mode='train')
    val_dataset = PcMapLocDataset(config, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                            shuffle=True, drop_last=True, 
                            num_workers=config['training']['num_workers'],
                            pin_memory=True, collate_fn=collate_fn_BEV)
    
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                          shuffle=True, drop_last=False,
                          num_workers=config['training']['num_workers'],
                          pin_memory=True, collate_fn=collate_fn_BEV)
                          
    print('Train dataset loaded, length:', len(train_loader.dataset))
    print('Validation dataset loaded, length:', len(val_loader.dataset))

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config)