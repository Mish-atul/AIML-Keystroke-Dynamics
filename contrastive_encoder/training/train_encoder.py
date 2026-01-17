"""
Encoder Training Module
=======================
Contrastive pretraining loop for the keystroke encoder.

Supports:
- NT-Xent loss (self-supervised)
- SupCon loss (supervised, default when labels available)
- Mixed precision training
- Checkpointing and early stopping
- WandB / TensorBoard logging
"""

import os
import time
from typing import Dict, Optional, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.encoder import KeystrokeEncoder, EncoderWithProjection, create_encoder
from ..models.losses import CombinedContrastiveLoss, NTXentLoss, SupConLoss
from ..data.dataset import ContrastiveKeystrokeDataset, create_dataloaders
from ..utils.helpers import (
    set_seed,
    get_device,
    save_json,
    AverageMeter,
    EarlyStopping,
    get_timestamp,
    count_parameters,
    format_number,
)
from ..config import EncoderConfig, get_default_config


class EncoderTrainer:
    """
    Trainer class for contrastive pretraining of keystroke encoder.
    """
    
    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        log_dir: str = "logs",
    ):
        """
        Args:
            config: Encoder configuration
            device: Compute device
            use_wandb: Whether to use Weights & Biases logging
            use_tensorboard: Whether to use TensorBoard logging
            log_dir: Directory for logs
        """
        self.config = config if config is not None else EncoderConfig()
        self.device = device if device is not None else get_device()
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        
        # Initialize logging
        self._init_logging()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None
        self.best_loss = float('inf')
        self.current_epoch = 0
    
    def _init_logging(self):
        """Initialize logging backends."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                print("TensorBoard not available, skipping")
        
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="keystroke-contrastive", config=vars(self.config))
            except ImportError:
                print("WandB not available, skipping")
                self.use_wandb = False
    
    def build_model(self) -> EncoderWithProjection:
        """Build encoder model with projection head."""
        model = create_encoder(
            input_features=3,
            seq_length=11,
            embedding_dim=self.config.embedding_dim,
            conv_channels=self.config.conv_channels,
            use_projection=True,
            projection_dim=self.config.embedding_dim,
        )
        
        model = model.to(self.device)
        
        param_count = count_parameters(model)
        print(f"Built encoder with {format_number(param_count)} parameters")
        
        return model
    
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build optimizer."""
        if self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        return optimizer
    
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01,
        )
        return scheduler
    
    def build_loss(self) -> nn.Module:
        """Build loss function."""
        if self.config.use_supcon:
            # Supervised contrastive as default
            loss_fn = CombinedContrastiveLoss(
                temperature=self.config.temperature,
                use_labels=True,
                supcon_weight=1.0,
                ntxent_weight=0.0,
            )
        else:
            # Self-supervised NT-Xent
            loss_fn = NTXentLoss(temperature=self.config.temperature)
        
        return loss_fn
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        scaler: Optional[GradScaler],
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        loss_meter = AverageMeter("loss")
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            labels = batch.get("label")
            
            if labels is not None:
                labels = labels.squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and scaler is not None:
                with autocast():
                    z1 = model(view1)
                    z2 = model(view2)
                    
                    if isinstance(loss_fn, CombinedContrastiveLoss):
                        loss = loss_fn(z1, z2, labels)
                    else:
                        loss = loss_fn(z1, z2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = model(view1)
                z2 = model(view2)
                
                if isinstance(loss_fn, CombinedContrastiveLoss):
                    loss = loss_fn(z1, z2, labels)
                else:
                    loss = loss_fn(z1, z2)
                
                loss.backward()
                optimizer.step()
            
            loss_meter.update(loss.item(), view1.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        
        return {"loss": loss_meter.avg}
    
    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Validate model."""
        model.eval()
        
        loss_meter = AverageMeter("loss")
        
        for batch in dataloader:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            labels = batch.get("label")
            
            if labels is not None:
                labels = labels.squeeze().to(self.device)
            
            z1 = model(view1)
            z2 = model(view2)
            
            if isinstance(loss_fn, CombinedContrastiveLoss):
                loss = loss_fn(z1, z2, labels)
            else:
                loss = loss_fn(z1, z2)
            
            loss_meter.update(loss.item(), view1.size(0))
        
        return {"loss": loss_meter.avg}
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        checkpoint_dir: str = "artifacts/checkpoints",
        save_path: str = "artifacts/encoder.pt",
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_data: Training data with 'X_seq' and 'y' keys
            val_data: Optional validation data
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
            checkpoint_dir: Directory for checkpoints
            save_path: Path to save final encoder
            
        Returns:
            Training history dictionary
        """
        epochs = epochs if epochs is not None else self.config.epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        
        if self.config.small_batch_mode:
            batch_size = self.config.small_batch_size
            print(f"Using small batch mode: batch_size={batch_size}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\n{'='*60}")
        print("ENCODER TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"Use AMP: {self.config.use_amp}")
        print(f"Use SupCon: {self.config.use_supcon}")
        print(f"Temperature: {self.config.temperature}")
        
        # Create dataloaders
        dataloaders = create_dataloaders(
            train_data,
            val_data,
            batch_size=batch_size,
            contrastive=True,
            seed=42,
        )
        
        train_loader = dataloaders["train"]
        val_loader = dataloaders.get("val")
        
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        
        # Build components
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer, epochs)
        self.loss_fn = self.build_loss()
        
        if self.config.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience,
            mode="min",
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "best_loss": float('inf'),
        }
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.loss_fn,
                self.scaler,
                epoch,
            )
            
            # Validate
            val_metrics = {"loss": float('inf')}
            if val_loader is not None:
                val_metrics = self.validate(self.model, val_loader, self.loss_fn)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["lr"].append(current_lr)
            
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("LR", current_lr, epoch)
            
            # WandB logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "lr": current_lr,
                    "epoch": epoch,
                })
            
            # Checkpointing
            current_loss = val_metrics["loss"] if val_loader else train_metrics["loss"]
            
            if current_loss < history["best_loss"]:
                history["best_loss"] = current_loss
                history["best_epoch"] = epoch
                self.best_loss = current_loss
                
                # Save best model
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "best_model.pt"),
                    epoch,
                    current_loss,
                )
            
            # Periodic checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
                    epoch,
                    current_loss,
                )
            
            # Early stopping (only if enabled)
            if not self.config.disable_early_stop and early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        history["total_time"] = total_time
        
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best loss: {history['best_loss']:.4f} at epoch {history['best_epoch']}")
        
        # Save final encoder (without projection head)
        self._save_encoder(save_path)
        
        # Save training history
        history_path = os.path.join(os.path.dirname(save_path), "training_history.json")
        save_json(history, history_path)
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        return history
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        loss: float,
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)
    
    def _save_encoder(self, path: str) -> None:
        """Save encoder weights (without projection head)."""
        # Extract encoder from EncoderWithProjection
        if isinstance(self.model, EncoderWithProjection):
            encoder = self.model.encoder
        else:
            encoder = self.model
        
        torch.save(encoder.state_dict(), path)
        print(f"Saved encoder to: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.model = self.build_model()
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("loss", float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_encoder(
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[EncoderConfig] = None,
    save_path: str = "artifacts/encoder.pt",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train encoder.
    
    Args:
        train_data: Training data with 'X_seq' and 'y' keys
        val_data: Optional validation data
        config: Encoder configuration
        save_path: Path to save encoder
        **kwargs: Additional arguments for Trainer
        
    Returns:
        Training history
    """
    trainer = EncoderTrainer(config=config, **kwargs)
    history = trainer.train(train_data, val_data, save_path=save_path)
    return history
