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

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():   
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 25, patience: int = 5, log_dir: str = 'runs', val_per_step: int = 1):
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs - 1}")):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            if step % val_per_step == 0:
                val_loss = validate_model(model, val_loader, criterion)
                print(f'Epoch {epoch}/{num_epochs - 1}, Step {step}, Validation Loss: {val_loss:.4f}')
                writer.add_scalar('Validation Loss', val_loss, epoch * len(train_loader) + step)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        
        val_loss = validate_model(model, val_loader, criterion)
        print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Validation Loss', val_loss, epoch)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()

if __name__ == "__main__":
    with open('conf/data/kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
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
    # print(len(train_loader.dataset))

    # training process
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config['training']['num_epochs'], patience=config['training']['early_stop']['patience'], log_dir=config['training']['log_dir'], val_per_step=config['training']['val_per_step_num'])