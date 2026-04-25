"""
ALS Digital Twin Training Script

Trains both classical baselines and deep learning models for:
1. Next-state prediction (ALSFRS-R scores)
2. Event survival prediction (death, gastrostomy, NIV, wheelchair, speech loss)

Usage:
    python train_digital_twin.py
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Local imports
from models.data_loader import (
    load_data, create_patient_splits, create_dataloaders,
    STATIC_FEATURES, TEMPORAL_FEATURES, ALSFRS_ITEMS
)
from models.digital_twin import ALSDigitalTwin, DigitalTwinLoss, create_model
from models.classical_baselines import ClassicalDigitalTwin


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
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'num_survival_intervals': 20,

    # Training
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 20,
    'patience': 7,
    'state_loss_weight': 1.0,
    'survival_loss_weight': 1.0,

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
}


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {k: 0 for k in ['state', 'survival', 'total']}
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
        state_pred, hazard_logits = model(
            static, temporal, mask, time_deltas, seq_lens
        )

        # Compute loss
        loss, losses = loss_fn(
            state_pred, hazard_logits,
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
            if k in loss_components:
                loss_components[k] += v.item()
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
    loss_components = {k: 0 for k in ['state', 'survival', 'total']}
    num_batches = 0

    # For metrics
    all_state_preds = []
    all_state_targets = []
    all_state_masks = []

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

            state_pred, hazard_logits = model(
                static, temporal, mask, time_deltas, seq_lens
            )

            loss, losses = loss_fn(
                state_pred, hazard_logits,
                state_target, state_mask,
                event_times, event_indicators
            )

            total_loss += loss.item()
            for k, v in losses.items():
                if k in loss_components:
                    loss_components[k] += v.item()
            num_batches += 1

            all_state_preds.append(state_pred.cpu())
            all_state_targets.append(state_target.cpu())
            all_state_masks.append(state_mask.cpu())

    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    # Compute MAE for state prediction
    state_preds = torch.cat(all_state_preds, dim=0)
    state_targets = torch.cat(all_state_targets, dim=0)
    state_masks = torch.cat(all_state_masks, dim=0)

    mae = ((state_preds - state_targets).abs() * state_masks).sum() / (state_masks.sum() + 1e-8)
    metrics = {'mae': mae.item()}

    return avg_loss, loss_components, metrics


def train_deep_model(train_loader, val_loader, train_dataset, config):
    """Train the deep learning digital twin model."""
    print("\n" + "=" * 60)
    print("Training Deep Learning Digital Twin")
    print("=" * 60)

    device = config['device']
    print(f"Using device: {device}")

    # Get feature dimensions
    static_dim = len([f for f in STATIC_FEATURES if f in train_dataset.static_features])
    temporal_dim = len(train_dataset.temporal_features)

    print(f"Static features: {static_dim}")
    print(f"Temporal features: {temporal_dim}")

    # Create model
    model_config = {
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'num_state_targets': 6,  # ALSFRS total + 5 domains
        'num_events': 5,  # death, gastrostomy, niv, wheelchair, speech
        'num_survival_intervals': config['num_survival_intervals'],
        'dropout': config['dropout']
    }

    model = create_model(static_dim, temporal_dim, model_config)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss function
    loss_fn = DigitalTwinLoss(
        state_weight=config['state_loss_weight'],
        survival_weight=config['survival_loss_weight'],
        num_events=5,
        num_intervals=config['num_survival_intervals']
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    print("\nStarting training...")
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # Validate
        val_loss, val_losses, val_metrics = validate(
            model, val_loader, loss_fn, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])

        # Print progress
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Train Loss: {train_loss:.4f} "
              f"(state: {train_losses['state']:.4f}, surv: {train_losses['survival']:.4f}) | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_metrics['mae']:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_config,
                'static_dim': static_dim,
                'temporal_dim': temporal_dim,
            }, output_dir / 'digital_twin_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(config['output_dir']) / 'training_history.csv', index=False)

    return model, history


def evaluate_model(model, test_loader, loss_fn, device):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("Evaluating Deep Learning Model")
    print("=" * 60)

    model.eval()

    all_state_preds = []
    all_state_targets = []
    all_state_masks = []
    all_event_times = []
    all_event_indicators = []
    all_median_preds = [[] for _ in range(5)]

    with torch.no_grad():
        for batch in test_loader:
            static = batch['static'].to(device)
            temporal = batch['temporal'].to(device)
            mask = batch['mask'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            seq_lens = batch['seq_lens'].to(device)

            # Get full predictions
            twin_state = model.get_digital_twin_state(
                static, temporal, mask, time_deltas, seq_lens
            )

            all_state_preds.append(twin_state['state_prediction'].cpu())
            all_state_targets.append(batch['state_target'])
            all_state_masks.append(batch['state_target_mask'])
            all_event_times.append(batch['event_times'])
            all_event_indicators.append(batch['event_indicators'])

            for i, med in enumerate(twin_state['median_event_times']):
                all_median_preds[i].append(med.cpu())

    # Concatenate
    state_preds = torch.cat(all_state_preds, dim=0)
    state_targets = torch.cat(all_state_targets, dim=0)
    state_masks = torch.cat(all_state_masks, dim=0)
    event_times = torch.cat(all_event_times, dim=0)
    event_indicators = torch.cat(all_event_indicators, dim=0)
    median_preds = [torch.cat(p, dim=0) for p in all_median_preds]

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
            actual_times = event_times[observed, i] * 365  # Convert back to days
            pred_times = median_preds[i][observed] * 365
            median_error = (pred_times - actual_times).abs().median().item()
            print(f"  {name}: Median Error = {median_error:.0f} days")


def train_classical_baselines(df_train, df_val, df_test, config):
    """Train classical baseline models."""
    print("\n" + "=" * 60)
    print("Training Classical Baseline Models")
    print("=" * 60)

    # Create and train model
    classical_twin = ClassicalDigitalTwin()
    classical_twin.fit(df_train)

    # Evaluate
    print("\nEvaluating classical baselines...")
    results = classical_twin.evaluate(df_test)

    print("\nState Prediction Results (Classical):")
    for target, metrics in results['state'].items():
        print(f"  {target}: MAE = {metrics['mae']:.2f}, R² = {metrics['r2']:.3f}")

    print("\nSurvival Prediction Results (Classical):")
    for event, metrics in results['survival'].items():
        metric_name = 'c_index' if 'c_index' in metrics else 'correlation'
        metric_val = metrics.get(metric_name, 'N/A')
        if isinstance(metric_val, float):
            print(f"  {event}: {metric_name} = {metric_val:.3f}")

    # Save model (pickle for sklearn models)
    import pickle
    output_dir = Path(config['output_dir'])
    with open(output_dir / 'classical_twin.pkl', 'wb') as f:
        pickle.dump(classical_twin, f)

    return classical_twin, results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("ALS Digital Twin Training Pipeline")
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

    # Save splits for reproducibility
    df_train.to_csv(output_dir / 'train_split.csv', index=False)
    df_val.to_csv(output_dir / 'val_split.csv', index=False)
    df_test.to_csv(output_dir / 'test_split.csv', index=False)

    # Train classical baselines
    classical_model, classical_results = train_classical_baselines(
        df_train, df_val, df_test, CONFIG
    )

    # Create dataloaders for deep model
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        df_train, df_val, df_test,
        batch_size=CONFIG['batch_size']
    )

    # Train deep model
    model, history = train_deep_model(
        train_loader, val_loader, train_dataset, CONFIG
    )

    # Load best model for evaluation
    checkpoint = torch.load(output_dir / 'digital_twin_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    device = CONFIG['device']
    loss_fn = DigitalTwinLoss(
        state_weight=CONFIG['state_loss_weight'],
        survival_weight=CONFIG['survival_loss_weight']
    ).to(device)

    evaluate_model(model, test_loader, loss_fn, device)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Models saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
