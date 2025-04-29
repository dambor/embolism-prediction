#!/usr/bin/env python
"""
Main entry point for running the embolism prediction model.

This script loads data, trains the model, evaluates performance,
and saves the results and visualizations.

Usage:
    python main.py [--data_dir DATA_DIR] [--max_patients MAX_PATIENTS] [--epochs EPOCHS]
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simplified_embolism_model directly
from simplified_embolism_model import EmbolismPredictor, EmbolismDataset, collate_fn

# Import visualization utilities
from utils_visualization import plot_training_history, create_performance_dashboard

# Import data generation function
from download_demo_data import generate_synthetic_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Embolism Prediction Model')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save/load data')
    parser.add_argument('--max_patients', type=int, default=200,
                        help='Maximum number of patients to include')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'data_path': args.data_dir,
        'ts_input_dim': 12,       # Number of time series features
        'static_input_dim': 9,    # Number of static features
        'hidden_dim': 64,         # Hidden dimension
        'dropout': 0.2,           # Dropout rate
        'learning_rate': 0.001,   # Learning rate
        'weight_decay': 1e-5,     # Weight decay
        'max_grad_norm': 1.0,     # Gradient clipping
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'pos_weight': 5.0,        # Positive class weight
        'max_patients': args.max_patients
    }
    
    # Check if data exists, otherwise generate it
    cohort_path = os.path.join(args.data_dir, 'cohort.csv')
    data_path = os.path.join(args.data_dir, 'preprocessed_data.csv')
    
    if not (os.path.exists(cohort_path) and os.path.exists(data_path)):
        print("Generating synthetic data...")
        generate_synthetic_data(args.data_dir)
    
    # Load data
    print("Loading and preprocessing data...")
    
    try:
        # Load preprocessed data
        data = pd.read_csv(data_path)
        cohort = pd.read_csv(cohort_path)
        
        # Convert datetime columns
        datetime_cols = ['charttime', 'chartdate']
        for col in datetime_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col])
        
        print(f"Loaded data with {len(data)} rows and {len(cohort)} patients")
        print(f"Positive cases (embolism): {cohort['embolism_type'].sum()}")
        print(f"Negative cases (no embolism): {len(cohort) - cohort['embolism_type'].sum()}")
        
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run download_demo_data.py first.")
        return
    
    # Create dataset
    dataset = EmbolismDataset(
        data=data,
        cohort=cohort,
        time_window=24,
        max_patients=config['max_patients']
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    print(f"Splitting data into train ({train_size}), validation ({val_size}), and test ({test_size}) sets")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create and train model
    print("Training model...")
    start_time = time.time()
    
    trainer = EmbolismPredictor(config)
    history = trainer.train(train_loader, val_loader, config['num_epochs'], patience=args.patience)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = trainer.predict(test_loader)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Training history
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Performance dashboard
    dashboard_fig = create_performance_dashboard(
        predictions['predictions'],
        predictions['labels'],
        history=history
    )
    dashboard_fig.savefig(os.path.join(args.output_dir, 'performance_dashboard.png'))
    
    # Save model
    model_path = os.path.join(args.output_dir, 'embolism_model.pt')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'test_metrics': test_metrics
    }, model_path)
    
    # Save results
    results = {
        'training_time': training_time,
        'test_loss': test_loss,
        'test_auroc': test_metrics['auroc'],
        'test_auprc': test_metrics['auprc'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'config': config
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()