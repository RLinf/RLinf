"""
Standalone script for training ResNet reward model.
Run with: python train_reward_model.py --config-name maniskill_train_reward_model
"""

import os
import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="maniskill_train_reward_model")
def main(cfg: DictConfig):
    """Main training function for reward model."""
    logger.info("Starting reward model training...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from rlinf.algorithms.rewards.embodiment.reward_model_trainer import RewardDataset
    from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel
    
    # Get training config
    train_cfg = cfg.reward_model_training
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading data from: {train_cfg.data_path}")
    dataset = RewardDataset(
        data_path=train_cfg.data_path,
        image_size=(train_cfg.image_size[1], train_cfg.image_size[2]),
        augment=True,
        max_success=train_cfg.max_success,
        max_failure=train_cfg.max_failure,
        use_last_frame=train_cfg.use_last_frame,
    )
    
    if len(dataset) == 0:
        logger.error("No data found! Please run data collection first.")
        return
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    val_size = int(len(dataset) * train_cfg.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model (ResNetRewardModel expects a DictConfig)
    resnet_cfg = cfg.reward.resnet
    model = ResNetRewardModel(resnet_cfg).to(device)
    
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Setup optimizer and loss
    # Note: Model already has Sigmoid in classifier, so use BCELoss not BCEWithLogitsLoss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    criterion = nn.BCELoss()
    
    # Setup save directory
    save_dir = train_cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {save_dir}")
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    # Epoch progress bar
    epoch_pbar = tqdm(range(train_cfg.epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Batch progress bar for training
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Train]",
            leave=False,
            unit="batch"
        )
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()  # Model already outputs probabilities
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update batch progress bar
            current_acc = train_correct / train_total
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Batch progress bar for validation
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Val]",
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device).float()
                
                outputs = model(images).squeeze(-1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()  # Model already outputs probabilities
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Update batch progress bar
                current_acc = val_correct / val_total
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}"
        )
        
        logger.info(
            f"Epoch {epoch+1}/{train_cfg.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if train_cfg.save_best:
                checkpoint_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model with val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= train_cfg.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
    
    logger.info(f"Training completed. Best val_acc: {best_val_acc:.4f}")
    logger.info(f"Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
