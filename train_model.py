"""
Training Script for Trading Transformer Model

This script handles the complete training pipeline:
1. Load CSV data for all 50 stocks
2. Preprocess and normalize data
3. Create training sequences with labels
4. Train transformer model with early stopping
5. Save trained model and preprocessing artifacts

Usage:
    python train_model.py --data_dir ./data --output_dir ./models --epochs 100

Author: Trading ML System
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Local imports
from transformer_model import TradingTransformer, ModelConfig, create_model
from utils import (
    load_multiple_stocks,
    get_feature_columns,
    handle_missing_values,
    fit_scalers,
    prepare_all_stocks,
    create_data_loaders,
    compute_class_weights,
    train_val_test_split,
    temporal_train_val_test_split,
    save_scaler,
    save_feature_columns,
    SIGNAL_NAMES
)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Trainer class for the Trading Transformer model.
    
    Handles training loop, validation, checkpointing, and metrics logging.
    """
    
    def __init__(
        self,
        model: TradingTransformer,
        config: ModelConfig,
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Training artifacts
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.class_weights = None
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_training(
        self,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Set up optimizer, scheduler, and loss function."""
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function with class weights
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"Training setup complete:")
        print(f"  - Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        print(f"  - Scheduler: ReduceLROnPlateau")
        if class_weights is not None:
            print(f"  - Class weights: {class_weights.numpy()}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            logits, _ = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        best_val_loss = float('inf')
        
        print(f"\nStarting training for up to {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("-" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                print(f"  -> New best model saved!")
            
            # Early stopping check
            if early_stopping(val_loss, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_score:.4f}")
                break
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("Training complete!")
        
        # Load best model for final metrics
        self.load_checkpoint(os.path.join(self.output_dir, 'best_model.pt'))
        val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
        
        print("\nFinal Validation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        
        # Classification report
        target_names = [SIGNAL_NAMES[i] for i in range(self.config.num_classes)]
        print("\nClassification Report:")
        print(classification_report(val_labels, val_preds, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        print("Confusion Matrix:")
        print(cm)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'best_epoch': early_stopping.best_epoch,
            'final_val_accuracy': val_acc
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
        
        return history
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'config': self.config.to_dict()
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")


def main(args):
    """Main training function."""
    
    print("=" * 60)
    print("Trading Transformer - Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model configuration
    config = ModelConfig(
        num_features=args.num_features,
        seq_length=args.seq_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_classes=3,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Step 1: Load data
    print("\n" + "-" * 60)
    print("Step 1: Loading stock data...")
    print("-" * 60)
    
    stocks_data = load_multiple_stocks(args.data_dir, pattern="*.csv")
    
    if len(stocks_data) == 0:
        print("ERROR: No stock data loaded. Check your data directory.")
        sys.exit(1)
    
    # Step 2: Determine feature columns
    print("\n" + "-" * 60)
    print("Step 2: Identifying feature columns...")
    print("-" * 60)
    
    # Get feature columns from first stock
    first_stock = list(stocks_data.values())[0]
    feature_columns = get_feature_columns(first_stock)
    
    # Limit to specified number of features if needed
    if len(feature_columns) > config.num_features:
        feature_columns = feature_columns[:config.num_features]
    elif len(feature_columns) < config.num_features:
        print(f"Warning: Only {len(feature_columns)} features found, expected {config.num_features}")
        config.num_features = len(feature_columns)
    
    print(f"Using {len(feature_columns)} features")
    
    # Save feature columns
    save_feature_columns(feature_columns, os.path.join(args.output_dir, 'feature_columns.pkl'))
    
    # Step 3: Handle missing values and prepare for scaling
    print("\n" + "-" * 60)
    print("Step 3: Preprocessing data...")
    print("-" * 60)
    
    # Clean data
    cleaned_stocks = {}
    for symbol, df in stocks_data.items():
        df_clean = handle_missing_values(df)
        if len(df_clean) > config.seq_length + 10:  # Need enough data for sequences
            cleaned_stocks[symbol] = df_clean
        else:
            print(f"  Skipping {symbol}: insufficient data after cleaning")
    
    print(f"Retained {len(cleaned_stocks)} stocks after cleaning")
    
    # Step 4: Fit scaler on all training data
    print("\n" + "-" * 60)
    print("Step 4: Fitting scaler...")
    print("-" * 60)
    
    scaler = fit_scalers(cleaned_stocks, feature_columns)
    save_scaler(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
    
    # Step 5: Create sequences for all stocks
    print("\n" + "-" * 60)
    print("Step 5: Creating sequences...")
    print("-" * 60)
    
    X_all, y_all = prepare_all_stocks(
        cleaned_stocks,
        feature_columns,
        scaler,
        seq_length=config.seq_length,
        close_column=args.close_column
    )
    
    print(f"\nTotal sequences: {len(X_all)}")
    print(f"Sequence shape: {X_all.shape}")
    print(f"Label distribution:")
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {SIGNAL_NAMES[u]}: {c} ({c/len(y_all)*100:.1f}%)")
    
    # Step 6: Split data
    print("\n" + "-" * 60)
    print("Step 6: Splitting data...")
    print("-" * 60)
    
    if args.temporal_split:
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
            X_all, y_all, train_ratio=0.7, val_ratio=0.15
        )
        print("Using temporal split (no shuffling)")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X_all, y_all, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        print("Using stratified random split")
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Save test data for later evaluation
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    print(f"Test data saved to {args.output_dir}")
    
    # Step 7: Create data loaders
    print("\n" + "-" * 60)
    print("Step 7: Creating data loaders...")
    print("-" * 60)
    
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=config.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Step 8: Compute class weights
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights.numpy()}")
    
    # Step 9: Create model and trainer
    print("\n" + "-" * 60)
    print("Step 9: Initializing model...")
    print("-" * 60)
    
    model = create_model(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, config, device, args.output_dir)
    trainer.setup_training(class_weights)
    
    # Step 10: Train model
    print("\n" + "-" * 60)
    print("Step 10: Training model...")
    print("-" * 60)
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience
    )
    
    # Step 11: Final evaluation on test set
    print("\n" + "-" * 60)
    print("Step 11: Final test set evaluation...")
    print("-" * 60)
    
    # Create test loader
    from utils import TradingDataset
    from torch.utils.data import DataLoader
    
    test_dataset = TradingDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    test_loss, test_acc, test_preds, test_labels = trainer.validate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    target_names = [SIGNAL_NAMES[i] for i in range(config.num_classes)]
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=target_names))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Models and artifacts saved to: {args.output_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trading Transformer Model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing stock CSV files')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained model and artifacts')
    parser.add_argument('--close_column', type=str, default='close',
                        help='Name of close price column in CSV')
    
    # Model architecture arguments
    parser.add_argument('--num_features', type=int, default=75,
                        help='Number of input features')
    parser.add_argument('--seq_length', type=int, default=40,
                        help='Sequence length for transformer input')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--temporal_split', action='store_true',
                        help='Use temporal split instead of random split')
    
    args = parser.parse_args()
    main(args)
