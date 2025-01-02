import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from loss import CircleLoss, CircleLossV2, CircleLossV3
# from network.network import Pcmapvpr
from network.nuscenes_network import BEVPcMapLoc
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn.functional as F
import random
import numpy as np
import time 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import pandas as pd
from nuscenes_dataloader import NuScenesDataset

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

# def save_experiment_para_results(results, config, filename):
#     results.update({
#         'learning_rate': config['training']['lr'],
#         'batch_size': config['training']['batch_size'],
#         'num_epochs': config['training']['num_epochs'],
#         'patience': config['training']['early_stop']['patience'],
#         'device': config['training']['device']
#     })
#     df = pd.DataFrame([results])
#     if os.path.exists(filename):
#         df.to_excel(filename, index=False, mode='a', header=False)
#     else:
#         df.to_excel(filename, index=False)

def get_top_percentage_indices(similarity_matrix, percentage=0.01):
    k = max(1, int(similarity_matrix.size(1) * percentage))
    return get_top_k_indices(similarity_matrix, k)

def get_top_k_ratio(top_k_indices, correct_index):
    return (top_k_indices == correct_index).any().item()

def calculate_geolocalization_metrics(top_indices, correct_coords, xy_list):
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
    return geo_1m, geo_5m, geo_10m

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        osm_feture_database = []
        pc_features = []
        xy_list = []   
        for (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_map, xy_torch) in tqdm(val_loader, desc="Validation"):
            imgs = imgs.to(device)
            trans = trans.to(device)
            rots = rots.to(device)
            intrins = intrins.to(device)
            post_trans = post_trans.to(device)
            post_rots = post_rots.to(device)
            lidar_data = lidar_data.to(device)
            lidar_mask = lidar_mask.to(device)
            car_trans = car_trans.to(device)
            yaw_pitch_roll = yaw_pitch_roll.to(device)
            osm_data = osm_map.to(device)
            xy = xy_torch.to(device)
            osm_descs, pc_bev_descs = model(osm_data, imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
            osm_feture_database.append(osm_descs)
            pc_features.append(pc_bev_descs)
            xy_list.append(xy)
            loss = criterion(osm_descs, pc_bev_descs)
            val_loss += loss.item() * osm_data.size(0)
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
    
    return results

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def send_email(subject, body, config):
    from_email = config['notification']['from_email']
    to_email = config['notification']['to_email']
    smtp_server = config['notification']['smtp_server']
    smtp_user = config['notification']['smtp_user']
    smtp_password = config['notification']['smtp_password']
    
    try:
        subject = Header(subject, 'utf-8').encode(0)
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ', '.join(to_email) if isinstance(to_email, list) else to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL(smtp_server, port=465) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print("Email sent successfully!")

    except smtplib.SMTPResponseException as e:
        error_code = e.smtp_code
        print(f"Failed to send email: {e}, error code: {error_code}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def format_geo_metrics(geo_metrics):
    table = "=== Geolocalization Metrics ===\n"
    table += "| Metric Type | 1m Ratio | 5m Ratio | 10m Ratio |\n"
    table += "|-------------|----------|----------|-----------|\n"
    
    for key, metrics in geo_metrics.items():
        table += f"| {key:<11} | {metrics['geo_1m_ratio']:.4f} | {metrics['geo_5m_ratio']:.4f} | {metrics['geo_10m_ratio']:.4f} |\n"
    
    return table

def format_training_results(val_loss, top1_ratio, top5_ratio, top1_percent_ratio, best_epoch, current_epoch, best_dict):
    table = "=== Training Results ===\n"
    table += "| Metric              | Value  |\n"
    table += "|---------------------|--------|\n"
    table += f"| Best Val Loss       | {val_loss:.4f} |\n"
    table += f"| Best Top-1 Ratio    | {top1_ratio:.4f} |\n"
    table += f"| Best Top-5 Ratio    | {top5_ratio:.4f} |\n"
    table += f"| Best Top-1% Ratio   | {top1_percent_ratio:.4f} |\n"
    table += f"| Best Epoch          | {best_epoch} |\n"
    table += f"| Current Epoch       | {current_epoch} |\n"
    
    table += "\n=== Best Results Timeline ===\n"
    table += "| Metric              | Epoch | Step  |\n"
    table += "|---------------------|-------|-------|\n"
    for key, value in best_dict.items():
        if key.startswith('best_'):
            metric_name = key.replace('best_', '').replace('_epoch', '')
            epoch = value
            step = best_dict.get(key.replace('_epoch', '_step'), 'N/A')
            table += f"| {metric_name:<19} | {epoch:<5} | {step:<5} |\n"
    
    return table

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, config: dict, 
                if_send_email: bool = True, if_early_stop: bool = True):
    num_epochs = config['training']['num_epochs']
    patience = config['training']['early_stop']['patience']
    log_dir = config['training']['log_dir']
    # val_per_step = config['training']['val_per_step_num']
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
    best_top1_ratio = 0.0  
    best_top5_ratio = 0.0  
    best_top_1_percent_ratio = 0.0  
    best_epoch = 0  
    best_geo_metrics = {
        'top1': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
        'top5': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0},
        'top1_percent': {'geo_1m_ratio': 0.0, 'geo_5m_ratio': 0.0, 'geo_10m_ratio': 0.0}
    }
    best_dict = {}
    model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_subdir)
    try:
        for epoch in range(num_epochs):
            # model.train()
            running_loss = 0.0
            for step, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_map, xy_torch) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs - 1}")):
                model.train()
                imgs = imgs.to(device)
                trans = trans.to(device)
                rots = rots.to(device)
                intrins = intrins.to(device)
                post_trans = post_trans.to(device)
                post_rots = post_rots.to(device)
                lidar_data = lidar_data.to(device)
                lidar_mask = lidar_mask.to(device)
                car_trans = car_trans.to(device)
                yaw_pitch_roll = yaw_pitch_roll.to(device)
                osm_data = osm_map.to(device)
                xy = xy_torch.to(device)

                optimizer.zero_grad()
                osm_descs, pc_bev_descs = model(osm_data, imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll) # [B, 4096] [B, 8192]
                # pad_size = 8192 - 256
                # osm_descs = F.pad(osm_descs, (0, pad_size), mode='constant', value=0)
                loss = criterion(osm_descs, pc_bev_descs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * osm_data.size(0)
                
                if step % 100 == 0:
                    step_loss = running_loss / ((step + 1) * train_loader.batch_size)
                    # print(f'Epoch {epoch}/{num_epochs - 1}, Step {step}, Training Loss: {step_loss:.4f}')
                    writer.add_scalar('Training Loss', step_loss, epoch * len(train_loader) + step)
                
                # if step % val_per_step == 0 and step != 0:
                #     val_results = validate_model(model, val_loader, criterion, device)
                #     val_loss = val_results['val_loss']
                #     if val_results['top_1_ratio'] > best_top1_ratio:
                #         best_top1_ratio = val_results['top_1_ratio']
                #         best_dict['best_top_1_epoch'] = epoch
                #         best_dict['best_top_1_step'] = step

                #     if val_results['top_5_ratio'] > best_top5_ratio:
                #         best_top5_ratio = val_results['top_5_ratio']
                #         best_dict['best_top_5_epoch'] = epoch
                #         best_dict['best_top_5_step'] = step

                #     if val_results['top_1_percent_ratio'] > best_top_1_percent_ratio:
                #         best_top_1_percent_ratio = val_results['top_1_percent_ratio']
                #         best_dict['best_top_1_percentage_epoch'] = epoch
                #         best_dict['best_top_1_percentage_step'] = step

                #     print(f'Epoch {epoch}/{num_epochs - 1}, Step {step}, Validation Loss: {val_loss:.4f}')
                #     writer.add_scalar('Validation Loss', val_loss, epoch * len(train_loader) + step)
                #     writer.add_scalar('Top 1 Ratio', val_results['top_1_ratio'], epoch * len(train_loader) + step)
                #     writer.add_scalar('Top 5 Ratio', val_results['top_5_ratio'], epoch * len(train_loader) + step)
                #     writer.add_scalar('Top 1% Ratio', val_results['top_1_percent_ratio'], epoch * len(train_loader) + step)
                    
                #     save_checkpoint({
                #         'epoch': epoch,
                #         'step': step,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'val_loss': val_loss,
                #         'top_1_ratio': val_results['top_1_ratio'],
                #         'top_5_ratio': val_results['top_5_ratio'],
                #         'top_1_percent_ratio': val_results['top_1_percent_ratio'],
                #     }, filename=f'{checkpoint_subdir}/checkpoint_epoch_{epoch}_step_{step}.pth.tar')
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
            writer.add_scalar('Training Loss', epoch_loss, epoch * len(train_loader))
            
            val_results = validate_model(model, val_loader, criterion, device)
            val_loss = val_results['val_loss']
            if val_results['top_1_ratio'] > best_top1_ratio:
                best_top1_ratio = val_results['top_1_ratio']
                best_dict['best_top_1_epoch'] = epoch
                best_dict['best_top_1_step'] = step

            if val_results['top_5_ratio'] > best_top5_ratio:
                best_top5_ratio = val_results['top_5_ratio']
                best_dict['best_top_5_epoch'] = epoch
                best_dict['best_top_5_step'] = step

            if val_results['top_1_percent_ratio'] > best_top_1_percent_ratio:
                best_top_1_percent_ratio = val_results['top_1_percent_ratio']
                best_dict['best_top_1_percentage_epoch'] = epoch
                best_dict['best_top_1_percentage_step'] = step

            print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')
            writer.add_scalar('Validation Loss', val_loss, epoch * len(train_loader) + step)
            writer.add_scalar('Top 1 Ratio', val_results['top_1_ratio'], epoch * len(train_loader) + step)
            writer.add_scalar('Top 5 Ratio', val_results['top_5_ratio'], epoch * len(train_loader) + step)
            writer.add_scalar('Top 1% Ratio', val_results['top_1_percent_ratio'], epoch * len(train_loader) + step)
            
            save_checkpoint({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'top_1_ratio': val_results['top_1_ratio'],
                'top_5_ratio': val_results['top_5_ratio'],
                'top_1_percent_ratio': val_results['top_1_percent_ratio'],
            }, filename=f'{checkpoint_subdir}/checkpoint_epoch_{epoch}.pth.tar')
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch  
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'top_1_ratio': val_results['top_1_ratio'],
                    'top_5_ratio': val_results['top_5_ratio'],
                    'top_1_percent_ratio': val_results['top_1_percent_ratio'],
                }, filename=f'{checkpoint_subdir}/best_checkpoint.pth.tar')
                best_dict['best_val_loss_epoch'] = epoch
                best_dict['best_val_loss_step'] = step
            
            if if_early_stop:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    if if_send_email:
                        email_body = "Training has been early stopped.\n\n This is Nuscenes with Circle Loss V2 with lr=10e-5 \n\n"
                        email_body += format_training_results(
                            best_val_loss, best_top1_ratio, best_top5_ratio, 
                            best_top_1_percent_ratio, best_epoch, epoch, best_dict
                        )
                        email_body += "\n" + format_geo_metrics(best_geo_metrics)
                        
                        send_email(
                            subject="Training Early Stopped",
                            body=email_body,
                            config=config
                        )
                    break
        
        # Training completed (either normally or through early stopping)
        writer.close()
        if if_send_email and not early_stopping.early_stop:
            email_body = "Training has completed successfully.\n\n"
            email_body += format_training_results(
                best_val_loss, best_top1_ratio, best_top5_ratio, 
                best_top_1_percent_ratio, best_epoch, epoch, best_dict
            )
            email_body += "\n" + format_geo_metrics(best_geo_metrics)
            
            send_email(
                subject="Training Completed",
                body=email_body,
                config=config
            )
    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        if if_send_email:
            error_message = f"Training failed with error: {str(e)}\n"
            error_message += f"Current epoch: {epoch}\n"
            error_message += f"Best validation loss: {best_val_loss:.4f}\n"
            error_message += f"Best epoch: {best_epoch}"
            
            send_email(
                subject="Training Failed",
                body=error_message,
                config=config
            )
        raise e

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    # load config
    with open('conf/data/nuscenes.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # set seed
    set_seed(seed = config['training']['seed'])

    print('Device:', config['training']['device'])
    # model, criterion, optimizer
    model = BEVPcMapLoc(config)
    from loss import ContrastiveLoss, CircleLossV4
    criterion = CircleLossV2()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # data loader
    train_dataset = NuScenesDataset(version='v1.0-trainval', dataroot='/data/Pcmaploc/data/Nuscenes', data_conf=config, is_train=True)  
    val_dataset = NuScenesDataset(version='v1.0-trainval', dataroot='/data/Pcmaploc/data/Nuscenes', data_conf=config, is_train=False) 
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True, num_workers=config['training']['num_workers'], pin_memory=True)
    print('Train dataset loaded, length:', len(train_loader.dataset))
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True, num_workers=config['training']['num_workers'], pin_memory=True)
    print('Validation dataset loaded, length:', len(val_loader.dataset))
    # print(val_loader.batch_size)
    # print(len(train_loader.dataset))

    # training process
    train_model(model, train_loader, val_loader, criterion, optimizer, config)

