"""
Train Transformer-based Autoencoder Digital Twin for ALS.

This script trains the Transformer Autoencoder model which learns a
compressed latent representation of patient disease trajectories.

Usage:
    python train_transformer_twin.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

from models.data_loader import (
    load_data, create_patient_splits, create_dataloaders,
    STATIC_FEATURES, TEMPORAL_FEATURES, ALSFRS_ITEMS
)
from models.transformer_twin import (
    TransformerAutoencoderTwin, TransformerTwinLoss, create_transformer_twin
)


# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # Data
    'data_dir': './cleaned_data',
    'output_dir': './trained_models',
    'test_size': 0.15,
    'val_size': 0.15,

    # Model architecture
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'latent_dim': 64,
    'dropout': 0.1,
    'use_variational': False,  # Set True for VAE-style training

    # Training
    'batch_size': 64,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 30,
    'patience': 10,
    'warmup_epochs': 3,

    # Loss weights
    'recon_weight': 1.0,
    'state_weight': 1.0,
    'survival_weight': 0.5,
    'kl_weight': 0.1,

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
}


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, warmup_epochs):
    """Train for one epoch with warmup."""
    model.train()
    total_loss = 0
    loss_components = {}
    num_batches = 0

    for batch in train_loader:
        # Move to device
        static = batch['static'].to(device)
        temporal = batch['temporal'].to(device)
        mask = batch['mask'].to(device)
        time_deltas = batch['time_deltas'].to(device)
        seq_lens = batch['seq_lens'].to(device)
        state_target = batch['state_target'].to(device)
        state_mask = batch['state_target_mask'].to(device)
        event_times = batch['event_times'].to(device)
        event_indicators = batch['event_indicators'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(static, temporal, mask, time_deltas, seq_lens)

        # Compute loss
        loss, losses = loss_fn(
            outputs,
            temporal, mask,
            state_target, state_mask,
            event_times, event_indicators
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item() if torch.is_tensor(v) else v
        num_batches += 1

    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


def validate(model, val_loader, loss_fn, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    loss_components = {}
    num_batches = 0

    all_state_preds = []
    all_state_targets = []
    all_state_masks = []
    all_latents = []

    with torch.no_grad():
        for batch in val_loader:
            static = batch['static'].to(device)
            temporal = batch['temporal'].to(device)
            mask = batch['mask'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            state_target = batch['state_target'].to(device)
            state_mask = batch['state_target_mask'].to(device)
            event_times = batch['event_times'].to(device)
            event_indicators = batch['event_indicators'].to(device)

            outputs = model(static, temporal, mask, time_deltas, seq_lens)

            loss, losses = loss_fn(
                outputs,
                temporal, mask,
                state_target, state_mask,
                event_times, event_indicators
            )

            total_loss += loss.item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v.item() if torch.is_tensor(v) else v
            num_batches += 1

            all_state_preds.append(outputs['state_pred'].cpu())
            all_state_targets.append(state_target.cpu())
            all_state_masks.append(state_mask.cpu())
            all_latents.append(outputs['latent'].cpu())

    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    # Compute metrics
    state_preds = torch.cat(all_state_preds, dim=0)
    state_targets = torch.cat(all_state_targets, dim=0)
    state_masks = torch.cat(all_state_masks, dim=0)
    latents = torch.cat(all_latents, dim=0)

    mae = ((state_preds - state_targets).abs() * state_masks).sum() / (state_masks.sum() + 1e-8)

    # Latent space stats
    latent_mean = latents.mean().item()
    latent_std = latents.std().item()

    metrics = {
        'mae': mae.item(),
        'latent_mean': latent_mean,
        'latent_std': latent_std
    }

    return avg_loss, loss_components, metrics


def evaluate_model(model, test_loader, device):
    """Comprehensive evaluation on test set."""
    print("\n" + "=" * 60)
    print("Evaluating Transformer Autoencoder Twin")
    print("=" * 60)

    model.eval()

    all_state_preds = []
    all_state_targets = []
    all_state_masks = []
    all_event_times = []
    all_event_indicators = []
    all_median_preds = [[] for _ in range(5)]
    all_latents = []
    all_reconstructions = []
    all_temporal = []

    with torch.no_grad():
        for batch in test_loader:
            static = batch['static'].to(device)
            temporal = batch['temporal'].to(device)
            mask = batch['mask'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            seq_lens = batch['seq_lens'].to(device)

            twin_state = model.get_digital_twin_state(
                static, temporal, mask, time_deltas, seq_lens
            )

            all_state_preds.append(twin_state['state_prediction'].cpu())
            all_state_targets.append(batch['state_target'])
            all_state_masks.append(batch['state_target_mask'])
            all_event_times.append(batch['event_times'])
            all_event_indicators.append(batch['event_indicators'])
            all_latents.append(twin_state['latent_state'].cpu())
            all_reconstructions.append(twin_state['reconstruction'].cpu())
            all_temporal.append(temporal.cpu())

            for i, med in enumerate(twin_state['median_event_times']):
                all_median_preds[i].append(med.cpu())

    # Concatenate
    state_preds = torch.cat(all_state_preds, dim=0)
    state_targets = torch.cat(all_state_targets, dim=0)
    state_masks = torch.cat(all_state_masks, dim=0)
    event_times = torch.cat(all_event_times, dim=0)
    event_indicators = torch.cat(all_event_indicators, dim=0)
    median_preds = [torch.cat(p, dim=0) for p in all_median_preds]
    latents = torch.cat(all_latents, dim=0)

    # Reconstruction error
    reconstructions = torch.cat(all_reconstructions, dim=0)
    temporal_data = torch.cat(all_temporal, dim=0)
    recon_error = (reconstructions - temporal_data).abs().mean().item()
    print(f"\nReconstruction MAE: {recon_error:.4f}")

    # Latent space statistics
    print(f"\nLatent Space Statistics:")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std: {latents.std():.4f}")
    print(f"  Min: {latents.min():.4f}")
    print(f"  Max: {latents.max():.4f}")

    # State prediction metrics
    print("\nState Prediction Results:")
    target_names = ['ALSFRS Total', 'Bulbar', 'Fine Motor', 'Gross Motor', 'Walking', 'Respiratory']
    for i, name in enumerate(target_names):
        mask_i = state_masks[:, i] > 0.5
        if mask_i.sum() > 0:
            mae = (state_preds[mask_i, i] - state_targets[mask_i, i]).abs().mean().item()
            print(f"  {name}: MAE = {mae:.2f}")

    # Event prediction metrics
    print("\nEvent Prediction Results (Median Time Error in Days):")
    event_names = ['Death', 'Gastrostomy', 'NIV', 'Wheelchair', 'Speech Loss']
    for i, name in enumerate(event_names):
        observed = event_indicators[:, i] > 0.5
        if observed.sum() > 10:
            actual_times = event_times[observed, i] * 365
            pred_times = median_preds[i][observed] * 365
            median_error = (pred_times - actual_times).abs().median().item()
            print(f"  {name}: Median Error = {median_error:.0f} days")

    return {
        'latents': latents.numpy(),
        'reconstruction_mae': recon_error
    }


def main():
    print("=" * 60)
    print("Transformer Autoencoder Digital Twin Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data(CONFIG['data_dir'])
    print(f"Loaded {len(df)} landmark instances from {df['SubjectUID'].nunique()} patients")

    # Create splits
    print("\nCreating train/val/test splits...")
    df_train, df_val, df_test = create_patient_splits(
        df,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        df_train, df_val, df_test,
        batch_size=CONFIG['batch_size']
    )

    # Get feature dimensions
    device = CONFIG['device']
    print(f"\nUsing device: {device}")

    static_dim = len([f for f in STATIC_FEATURES if f in train_dataset.static_features])
    temporal_dim = len(train_dataset.temporal_features)

    print(f"Static features: {static_dim}")
    print(f"Temporal features: {temporal_dim}")

    # Create model
    model_config = {
        'd_model': CONFIG['d_model'],
        'n_heads': CONFIG['n_heads'],
        'n_layers': CONFIG['n_layers'],
        'latent_dim': CONFIG['latent_dim'],
        'num_state_targets': 6,
        'num_events': 5,
        'num_survival_intervals': 20,
        'dropout': CONFIG['dropout'],
        'use_variational': CONFIG['use_variational']
    }

    model = create_transformer_twin(static_dim, temporal_dim, model_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss function
    loss_fn = TransformerTwinLoss(
        recon_weight=CONFIG['recon_weight'],
        state_weight=CONFIG['state_weight'],
        survival_weight=CONFIG['survival_weight'],
        kl_weight=CONFIG['kl_weight'],
        num_events=5,
        num_intervals=20
    ).to(device)

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 'val_mae': [],
        'recon_loss': [], 'state_loss': [], 'survival_loss': []
    }

    print("\nStarting training...")
    print("-" * 80)

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            epoch, CONFIG['warmup_epochs']
        )

        # Validate
        val_loss, val_losses, val_metrics = validate(
            model, val_loader, loss_fn, device
        )

        # Update scheduler
        scheduler.step()

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['recon_loss'].append(val_losses.get('reconstruction', 0))
        history['state_loss'].append(val_losses.get('state', 0))
        history['survival_loss'].append(val_losses.get('survival', 0))

        # Print progress
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:2d}/{CONFIG['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MAE: {val_metrics['mae']:.2f} | "
              f"Recon: {val_losses.get('reconstruction', 0):.4f} | "
              f"LR: {lr:.2e}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_config,
                'static_dim': static_dim,
                'temporal_dim': temporal_dim,
            }, output_dir / 'transformer_twin_best.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print("-" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'transformer_training_history.csv', index=False)

    # Load best model for evaluation
    checkpoint = torch.load(output_dir / 'transformer_twin_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    eval_results = evaluate_model(model, test_loader, device)

    # Save latent representations for analysis
    np.save(output_dir / 'test_latents.npy', eval_results['latents'])

    # Save config
    with open(output_dir / 'transformer_config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Models saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
