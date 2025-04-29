"""
Visualization utilities for embolism prediction model results.

This module provides functions for visualizing model performance,
training history, and feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix


def plot_roc_curve(fpr, tpr, auroc, ax=None):
    """
    Plot Receiver Operating Characteristic (ROC) curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'AUROC = {auroc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Add labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    return ax


def plot_pr_curve(precision, recall, auprc, ax=None):
    """
    Plot Precision-Recall curve
    
    Args:
        precision: Precision values
        recall: Recall values
        auprc: Area under precision-recall curve
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Precision-Recall curve
    ax.plot(recall, precision, lw=2, label=f'AUPRC = {auprc:.3f}')
    
    # Add labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return ax


def plot_training_history(history, figsize=(12, 5)):
    """
    Plot training history
    
    Args:
        history: Dictionary of training history
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    axes[1].plot(history['val_auroc'], label='AUROC')
    axes[1].plot(history['val_auprc'], label='AUPRC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_performance_dashboard(predictions, labels, history=None, figsize=(14, 10)):
    """
    Create comprehensive performance dashboard
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        history: Training history (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate performance metrics
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auroc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    auprc = auc(recall, precision)
    
    # Create optimal threshold predictions
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    binary_preds = (predictions >= optimal_threshold).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(labels, binary_preds)
    
    # Create figure
    if history:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, auroc, ax=axes[0])
    
    # Plot PR curve
    plot_pr_curve(precision, recall, auprc, ax=axes[1])
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=axes[2]
    )
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')
    axes[2].set_title('Confusion Matrix')
    
    # Plot training history if provided
    if history:
        # Plot loss
        axes[3].plot(history['train_loss'], label='Train Loss')
        axes[3].plot(history['val_loss'], label='Validation Loss')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Training and Validation Loss')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
