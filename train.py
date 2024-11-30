import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from loss import CircleLoss
from network.network import Pcmapvpr
from kitti_dataloader import PcMapLocDataset
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn.functional as F
import random
import numpy as np
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        self.best_loss = val_loss

def get_top_k_indices(similarity_matrix, k):
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    return top_k_indices

def get_top_percentage_indices(similarity_matrix, percentage=0.01):
    k = int(similarity_matrix.size(1) * percentage)
    return get_top_k_indices(similarity_matrix, k)

def get_top_k_ratio(similarity_matrix, top_k_indices):
    correct_matches = (top_k_indices == torch.arange(similarity_matrix.size(0), device=similarity_matrix.device).unsqueeze(1)).sum().item()
    return correct_matches / similarity_matrix.size(0)

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        osm_feture_database = []
        pc_features = []   
        for data in tqdm(val_loader, desc="Validation"):
            osm_data, pc_bev_data = data['osm_map'].to(device), data['pc_bev_img'].to(device)
            osm_descs, pc_bev_descs = model(osm_data, pc_bev_data)
            osm_feture_database.append(osm_descs)
            pc_features.append(pc_bev_descs)
            loss = criterion(osm_descs, pc_bev_descs)
            val_loss += loss.item() * osm_data.size(0)
        osm_feture_database = torch.cat(osm_feture_database, dim=0)
        pc_features = torch.cat(pc_features, dim=0)
        
        # Calculate cosine similarity
        similarity_matrix = F.cosine_similarity(pc_features.unsqueeze(1), osm_feture_database.unsqueeze(0), dim=2)
        
        # Get top k and top percentage k indices
        top1_indices = get_top_k_indices(similarity_matrix, k=1)
        top_k_indices = get_top_k_indices(similarity_matrix, k=5)
        top_percentage_indices = get_top_percentage_indices(similarity_matrix, percentage=0.01)
        
        # Calculate ratios
        top1_ratio = get_top_k_ratio(similarity_matrix, top1_indices)
        top_k_ratio = get_top_k_ratio(similarity_matrix, top_k_indices)
        top_percentage_ratio = get_top_k_ratio(similarity_matrix, top_percentage_indices)
        
    val_loss /= len(val_loader.dataset)
    
    results = {
        'val_loss': val_loss,
        'top1_ratio': top1_ratio,
        'top_k_ratio': top_k_ratio,
        'top_percentage_ratio': top_percentage_ratio
    }
    
    return results

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, config: dict):
    num_epochs = config['training']['num_epochs']
    patience = config['training']['early_stop']['patience']
    log_dir = config['training']['log_dir']
    val_per_step = config['training']['val_per_step_num']
    device = torch.device(config['training']['device'])
    checkpoint_dir = config['training']['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_subdir = os.path.join(checkpoint_dir, current_time)
    os.makedirs(checkpoint_subdir)
    
    log_subdir = os.path.join(log_dir, current_time)
    os.makedirs(log_subdir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_subdir)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs - 1}")):
            osm_data, pc_bev_data = data['osm_map'].to(device), data['pc_bev_img'].to(device)
            optimizer.zero_grad()
            osm_descs, pc_bev_descs = model(osm_data, pc_bev_data) # [B, 4096] [B, 8192]
            loss = criterion(osm_descs, pc_bev_descs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * osm_data.size(0)
            
            if step % val_per_step == 0:
                val_results = validate_model(model, val_loader, criterion, device)
                val_loss = val_results['val_loss']
                print(f'Epoch {epoch}/{num_epochs - 1}, Step {step}, Validation Loss: {val_loss:.4f}')
                writer.add_scalar('Validation Loss', val_loss, epoch * len(train_loader) + step)
                writer.add_scalar('Top1 Ratio', val_results['top1_ratio'], epoch * len(train_loader) + step)
                writer.add_scalar('Top K Ratio', val_results['top_k_ratio'], epoch * len(train_loader) + step)
                writer.add_scalar('Top Percentage Ratio', val_results['top_percentage_ratio'], epoch * len(train_loader) + step)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        
        val_results = validate_model(model, val_loader, criterion, device)
        val_loss = val_results['val_loss']
        print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Top1 Ratio', val_results['top1_ratio'], epoch)
        writer.add_scalar('Top K Ratio', val_results['top_k_ratio'], epoch)
        writer.add_scalar('Top Percentage Ratio', val_results['top_percentage_ratio'], epoch)
        
        # Save current epoch checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'top1_ratio': val_results['top1_ratio'],
            'top_k_ratio': val_results['top_k_ratio'],
            'top_percentage_ratio': val_results['top_percentage_ratio'],
        }, filename=f'{checkpoint_subdir}/checkpoint_epoch_{epoch}.pth.tar')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'top1_ratio': val_results['top1_ratio'],
                'top_k_ratio': val_results['top_k_ratio'],
                'top_percentage_ratio': val_results['top_percentage_ratio'],
            }, filename=f'{checkpoint_subdir}/best_checkpoint.pth.tar')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()

def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# 设置种子
seed = 42
set_seed(seed)

if __name__ == "__main__":
    # load config
    with open('conf/data/kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # set seed
    set_seed(seed = config['training']['seed'],device=config['training']['device'])

    # model, criterion, optimizer
    model = Pcmapvpr(config) 
    criterion = CircleLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # data loader
    train_dataset = PcMapLocDataset(config, 'train')  
    val_dataset = PcMapLocDataset(config, 'val') 
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    print('Train dataset loaded, length:', len(train_loader.dataset))
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True)
    print('Validation dataset loaded, length:', len(val_loader.dataset))
    # print(val_loader.batch_size)
    # print(len(train_loader.dataset))

    # training process
    train_model(model, train_loader, val_loader, criterion, optimizer, config)