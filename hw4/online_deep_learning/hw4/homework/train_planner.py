"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 100 --lr 1e-3
    python3 -m homework.train_planner --model transformer_planner --epochs 100 --lr 1e-3
    python3 -m homework.train_planner --model vit_planner --epochs 100 --lr 1e-3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from .models import MLPPlanner, TransformerPlanner, ViTPlanner, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    metric = PlannerMetric()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Different models expect different inputs
        if isinstance(model, (MLPPlanner, TransformerPlanner)):
            # Models that use track boundaries
            preds = model(track_left=batch['track_left'], track_right=batch['track_right'])
        else:  # ViTPlanner
            # Model that uses images
            preds = model(image=batch['image'])
        
        # Compute loss
        waypoints = batch['waypoints']
        waypoints_mask = batch['waypoints_mask']
        
        # Only compute loss on valid waypoints
        if waypoints_mask.sum() > 0:  # Make sure there are valid waypoints
            loss = criterion(preds[waypoints_mask], waypoints[waypoints_mask])
        else:
            # Skip this batch if no valid waypoints
            continue
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update metrics
        metric.add(preds, waypoints, waypoints_mask)
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    metrics = metric.compute()
    
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch"""
    model.eval()
    total_loss = 0.0
    metric = PlannerMetric()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            if isinstance(model, (MLPPlanner, TransformerPlanner)):
                preds = model(track_left=batch['track_left'], track_right=batch['track_right'])
            else:  # ViTPlanner
                preds = model(image=batch['image'])
            
            # Compute loss
            waypoints = batch['waypoints']
            waypoints_mask = batch['waypoints_mask']
            
            if waypoints_mask.sum() > 0:  # Make sure there are valid waypoints
                loss = criterion(preds[waypoints_mask], waypoints[waypoints_mask])
                total_loss += loss.item()
                
                # Update metrics
                metric.add(preds, waypoints, waypoints_mask)
    
    avg_loss = total_loss / len(dataloader)
    metrics = metric.compute()
    
    return avg_loss, metrics


def test_model_forward(model, device, transform_pipeline):
    """Test if model forward pass works correctly"""
    print("Testing model forward pass...")
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, (MLPPlanner, TransformerPlanner)):
            # Test with dummy track data
            batch_size = 2
            n_track = 10
            track_left = torch.randn(batch_size, n_track, 2).to(device)
            track_right = torch.randn(batch_size, n_track, 2).to(device)
            
            try:
                output = model(track_left=track_left, track_right=track_right)
                expected_shape = (batch_size, 3, 2)
                assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
                print(f"✓ Model output shape: {output.shape}")
                return True
            except Exception as e:
                print(f"✗ Model forward pass failed: {e}")
                return False
        else:  # ViTPlanner
            # Test with dummy image data
            batch_size = 2
            image = torch.randn(batch_size, 3, 96, 128).to(device)
            
            try:
                output = model(image=image)
                expected_shape = (batch_size, 3, 2)
                assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
                print(f"✓ Model output shape: {output.shape}")
                return True
            except Exception as e:
                print(f"✗ Model forward pass failed: {e}")
                return False


def main():
    parser = argparse.ArgumentParser(description='Train planner models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mlp_planner', 'transformer_planner', 'vit_planner'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--data_path', type=str, default='drive_data', help='Path to data')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Training {args.model} on {device}")
    
    # Create model
    if args.model == 'mlp_planner':
        model = MLPPlanner()
        transform_pipeline = 'state_only'  # MLP only needs track boundaries
    elif args.model == 'transformer_planner':
        model = TransformerPlanner()
        transform_pipeline = 'state_only'  # Transformer only needs track boundaries
    elif args.model == 'vit_planner':
        model = ViTPlanner()
        transform_pipeline = 'default'  # ViT needs images
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Test model forward pass
    if not test_model_forward(model, device, transform_pipeline):
        print("Model test failed. Exiting...")
        return
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # MSE loss for regression
    
    # Load data
    try:
        train_loader = load_data(
            dataset_path=f"{args.data_path}/train",
            transform_pipeline=transform_pipeline,
            return_dataloader=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_loader = load_data(
            dataset_path=f"{args.data_path}/val",
            transform_pipeline=transform_pipeline,
            return_dataloader=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure {args.data_path} exists and contains train/ and val/ directories")
        return
    
    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Longitudinal Error: {train_metrics['longitudinal_error']:.4f}")
            print(f"Train Lateral Error: {train_metrics['lateral_error']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Longitudinal Error: {val_metrics['longitudinal_error']:.4f}")
            print(f"Val Lateral Error: {val_metrics['lateral_error']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = save_model(model)
                print(f"Saved best model to {model_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
