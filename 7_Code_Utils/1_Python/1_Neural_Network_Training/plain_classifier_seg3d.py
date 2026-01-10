import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import toml
import wandb
from tqdm import tqdm
import nibabel as nib
from collections import defaultdict
import matplotlib.pyplot as plt

class SimpleSegTrainer:
    """
    Universal Trainer Framework
    """
    
    def __init__(self, config_path: str):
        # read TOML file
        with open(config_path, 'r') as f:
            self.cfg = toml.load(f)
        
        # setting training device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # main components
        self.network: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: Any = None
        self.loss_fn: Callable = None
        
        # training status 
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # model checkpoint saving
        self.save_dir = Path(self.cfg['logging']['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        # logging
        if self.cfg['logging']['use_wandb']:
            wandb.init(project=self.cfg['logging']['project_name'], 
                      config=self.cfg, name=self.cfg['logging']['run_name'])
    
    def set_network(self, network: nn.Module):
        """add neural network"""
        self.network = network.to(self.device)
        print(f"Network Params: {sum(p.numel() for p in network.parameters()):,}")
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """add optimizer"""
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler: Any):
        """add LR scheduler"""
        self.scheduler = scheduler
    
    def set_loss_fn(self, loss_fn: Callable):
        """add loss function"""
        self.loss_fn = loss_fn
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """train ONE epoch"""
        self.network.train()
        metrics = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {self.epoch}")
        for batch in pbar:
            # move training data and label to device
            image = batch['image'].to(self.device, dtype=torch.float32)  # (B, 1, D, H, W)
            target = batch['target'].to(self.device, dtype=torch.long)    # (B, D, H, W)
            
            # prediction
            self.optimizer.zero_grad()
            pred = self.network(image)  # (B, num_classes, D, H, W)
            
            # compute loss
            loss = self.loss_fn(pred, target)
            
            # backward pass
            loss.backward()
            if 'grad_clip' in self.cfg['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.cfg['training']['grad_clip']
                )
            self.optimizer.step()
            
            # 指标
            metrics['loss'].append(loss.item())
            with torch.no_grad():
                dice = self._compute_dice(pred, target)
                metrics['dice'].append(dice)
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice})
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        test or validate on validation set
        """
        self.network.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Val Epoch {self.epoch}")
            for batch in pbar:
                image = batch['image'].to(self.device, dtype=torch.float32)
                target = batch['target'].to(self.device, dtype=torch.long)
                
                # forward pass
                pred = self.network(image)
                
                # compute loss
                loss = self.loss_fn(pred, target)
                metrics['loss'].append(loss.item())
                
                # compute dice score
                dice = self._compute_dice(pred, target)
                metrics['dice'].append(dice)
                
                pbar.set_postfix({'loss': loss.item(), 'dice': dice})
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        compute DICE score
        """
        num_classes = pred.shape[1]
        pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        dice_scores = []
        for cls in range(1, num_classes):  # 跳过背景
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice.item())
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        main training loop
        """
        
        print(f"Starting training [{self.cfg['training']['num_epochs']}] epochs")
        
        for epoch in range(self.cfg['training']['num_epochs']):
            self.epoch = epoch
            
            # train ONE epoch
            train_metrics = self.train_epoch(train_loader)
            
            # validation
            val_metrics = self.validate(val_loader)
            
            # update LR scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # add logging info
            self._log_metrics({**train_metrics, **val_metrics})
            
            # save checkpoints
            self._save_checkpoint(val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """
        logging
        """
        print(f"Epoch {self.epoch}: " + 
              f"Train Loss: {metrics['loss']:.4f}, " +
              f"Val Dice: {metrics['dice']:.4f}")
        
        if self.cfg['logging']['use_wandb']:
            wandb.log(metrics, step=self.epoch)
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """
        save latest & best model
        """
        current_dice = val_metrics['dice']
        
        # save latest
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_dice': current_dice,
            'config': self.cfg
        }, self.save_dir / 'last.ckpt')
        
        # save best
        if current_dice > self.best_metric:
            self.best_metric = current_dice
            self.patience_counter = 0
            
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.network.state_dict(),
                'val_dice': current_dice,
            }, self.save_dir / 'best.ckpt')
            
            print(f"  New best model! Dice: {current_dice:.4f}")
        else:
            self.patience_counter += 1
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """check for early stopping"""
        patience = self.cfg['training'].get('patience', None)
        if patience and self.patience_counter >= patience:
            return True
        return False
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """predict on testing set"""
        self.network.eval()
        with torch.no_grad():
            image = image.to(self.device, dtype=torch.float32)
            pred = self.network(image)
            return torch.argmax(pred, dim=1)  # 返回类别索引