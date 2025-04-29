"""
Simplified Embolism Detection Model

This module contains all the necessary components for the embolism prediction model:
- Time Series Encoder
- Embolism Predictor Model
- Dataset implementation
- Training and evaluation utilities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix


#----------------
# 1. TIME SERIES ENCODER
#----------------

class TimeSeriesEncoder(nn.Module):
    """
    LSTM-based encoder for vital signs and lab values
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        """
        Initialize the time series encoder
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            dropout: Dropout probability
        """
        super(TimeSeriesEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,  # Dropout is applied separately
            bidirectional=False
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        """
        Encode time series data
        
        Args:
            x: Time series data tensor of shape (batch_size, seq_len, input_dim)
            lengths: Lengths of sequences in the batch
            
        Returns:
            Tensor of encoded time series
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Pack padded sequences
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        outputs, (hidden, _) = self.lstm(packed_x)
        
        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Apply attention
        attention_scores = self.attention(outputs)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Compute weighted sum
        context = torch.sum(outputs * attention_weights, dim=1)
        
        return context


#----------------
# 2. EMBOLISM PREDICTION MODEL
#----------------

class EmbolismPredictor:
    """
    Model for embolism prediction
    """
    
    def __init__(self, config):
        """
        Initialize the embolism predictor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = EmbolismModel(
            ts_input_dim=config['ts_input_dim'],
            static_input_dim=config['static_input_dim'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create loss function with class weights
        pos_weight = torch.tensor([config['pos_weight']]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Initialize best metrics
        self.best_auroc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train(self, train_loader, val_loader, num_epochs, patience=5):
        """
        Train the model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary of training history
        """
        print(f"Starting training for {num_epochs} epochs with early stopping (patience={patience})")
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'val_auprc': []
        }
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move data to device
                ts_data = batch['ts_data'].to(self.device)
                ts_lengths = batch['ts_lengths'].to(self.device)
                static_data = batch['static_data'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(ts_data, ts_lengths, static_data)
                
                # FIX: Ensure output and label have the same dimensions
                # Make sure outputs and labels have the same dimensions
                if outputs.shape != labels.shape:
                    # Reshape labels to match outputs if needed
                    labels = labels.view(outputs.shape)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                
                # Optimize
                self.optimizer.step()
                
                # Update loss
                train_loss += loss.item() * ts_data.size(0)
            
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_auroc'].append(val_metrics['auroc'])
            history['val_auprc'].append(val_metrics['auprc'])
            
            # Early stopping
            if val_metrics['auroc'] > self.best_auroc:
                self.best_auroc = val_metrics['auroc']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_model('best_model.pt')
                improved = "(improved)"
            else:
                self.patience_counter += 1
                improved = ""
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val AUROC: {val_metrics["auroc"]:.4f} {improved} | '
                  f'Val AUPRC: {val_metrics["auprc"]:.4f}')
            
            # Check early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch was {self.best_epoch+1}.")
                break
        
        # Load best model
        self.load_model('best_model.pt')
        
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of loss and metrics dictionary
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                ts_data = batch['ts_data'].to(self.device)
                ts_lengths = batch['ts_lengths'].to(self.device)
                static_data = batch['static_data'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(ts_data, ts_lengths, static_data)
                
                # FIX: Ensure output and label have the same dimensions
                if outputs.shape != labels.shape:
                    # Reshape labels to match outputs if needed
                    labels = labels.view(outputs.shape)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Update loss
                total_loss += loss.item() * ts_data.size(0)
                
                # Save predictions and labels
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Check if we have both classes represented
        if len(np.unique(all_labels)) < 2:
            print("Warning: Only one class present in labels, ROC calculation skipped")
            metrics = {
                'auroc': 0.5,
                'auprc': 0.5,
                'sensitivity': 0.0 if all_labels[0] == 0 else 1.0,
                'specificity': 1.0 if all_labels[0] == 0 else 0.0
            }
            return avg_loss, metrics
        
        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        auroc = auc(fpr, tpr)
        
        # Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        auprc = auc(recall, precision)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate additional metrics at optimal threshold
        binary_preds = (all_preds >= optimal_threshold).astype(int)
        
        # Handle edge cases where we might have all of one class
        if len(np.unique(binary_preds)) < 2:
            sensitivity = 0.0 if binary_preds[0] == 0 else 1.0
            specificity = 1.0 if binary_preds[0] == 0 else 0.0
        else:
            tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'auroc': auroc,
            'auprc': auprc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        return avg_loss, metrics
    
    def predict(self, data_loader):
        """
        Generate predictions
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dictionary of predictions and ground truth
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_subject_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                ts_data = batch['ts_data'].to(self.device)
                ts_lengths = batch['ts_lengths'].to(self.device)
                static_data = batch['static_data'].to(self.device)
                
                # Forward pass
                outputs = self.model(ts_data, ts_lengths, static_data)
                
                # Save predictions and metadata
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(batch['labels'].squeeze().cpu().numpy())
                all_subject_ids.extend(batch['subject_ids'])
        
        results = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'subject_ids': all_subject_ids
        }
        
        return results
    
    def save_model(self, path):
        """
        Save model checkpoint
        
        Args:
            path: Path to save the model
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_auroc': self.best_auroc,
                'best_epoch': self.best_epoch,
                'config': self.config
            }, path)
        except Exception as e:
            print(f"Warning: Could not save model due to: {e}")
            print("Continuing without saving...")
        
    def load_model(self, path):
        """
        Load model checkpoint
        
        Args:
            path: Path to load the model from
        """
        try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_auroc = checkpoint['best_auroc']
                self.best_epoch = checkpoint['best_epoch']
                print(f"Loaded model from {path} (best AUROC: {self.best_auroc:.4f}, epoch: {self.best_epoch+1})")
            else:
                print(f"Warning: No model found at {path}")
        except Exception as e:
            print(f"Warning: Could not load model due to: {e}")
            print("Continuing with current model state...")


class EmbolismModel(nn.Module):
    """
    Neural network model for embolism prediction
    """
    
    def __init__(self, ts_input_dim, static_input_dim, hidden_dim=64, dropout=0.2):
        """
        Initialize the model
        
        Args:
            ts_input_dim: Dimension of time series input features
            static_input_dim: Dimension of static input features
            hidden_dim: Dimension of hidden state
            dropout: Dropout probability
        """
        super(EmbolismModel, self).__init__()
        
        # Time series encoder
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        fusion_dim = hidden_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, ts_data, ts_lengths, static_data):
        """
        Forward pass
        
        Args:
            ts_data: Time series data tensor
            ts_lengths: Lengths of time series sequences
            static_data: Static feature tensor
            
        Returns:
            Embolism prediction logits
        """
        # Encode time series data
        ts_embeddings = self.ts_encoder(ts_data, ts_lengths)
        
        # Encode static features
        static_embeddings = self.static_encoder(static_data)
        
        # Concatenate all modalities
        combined = torch.cat([
            ts_embeddings,
            static_embeddings
        ], dim=1)
        
        # Fuse features
        fused = self.fusion_layer(combined)
        
        # Generate prediction
        logits = self.output_layer(fused)
        
        return logits


#----------------
# 3. DATASET
#----------------

class EmbolismDataset(Dataset):
    """
    Dataset for embolism prediction
    """
    
    def __init__(self, data, cohort, time_window=24, max_patients=None):
        """
        Initialize the dataset
        
        Args:
            data: DataFrame containing preprocessed data
            cohort: DataFrame containing patient cohort information
            time_window: Time window in hours for input data
            max_patients: Maximum number of patients to include
        """
        # Limit the number of patients if specified
        if max_patients is not None and len(cohort['subject_id'].unique()) > max_patients:
            # Ensure a mix of positive and negative cases
            pos_subjects = cohort[cohort['embolism_type'] > 0]['subject_id'].unique()
            neg_subjects = cohort[cohort['embolism_type'] == 0]['subject_id'].unique()
            
            # Take a balanced sample
            pos_sample_size = min(int(max_patients * 0.3), len(pos_subjects))
            neg_sample_size = min(max_patients - pos_sample_size, len(neg_subjects))
            
            sampled_pos = np.random.choice(pos_subjects, size=pos_sample_size, replace=False)
            sampled_neg = np.random.choice(neg_subjects, size=neg_sample_size, replace=False)
            
            sampled_subjects = np.concatenate([sampled_pos, sampled_neg])
            cohort = cohort[cohort['subject_id'].isin(sampled_subjects)]
            data = data[data['subject_id'].isin(sampled_subjects)]
        
        self.data = data
        self.cohort = cohort
        self.time_window = time_window
        
        # Create sample windows
        self.windows = self._create_windows()
        
        print(f"Created dataset with {len(self.windows)} windows from {len(np.unique(cohort['subject_id']))} patients")
        
    def _create_windows(self):
        """
        Create sample windows for training and prediction
        
        Returns:
            List of window dictionaries
        """
        windows = []
        
        # Group by patient
        for subject_id, subject_group in self.data.groupby('subject_id'):
            # Get patient info
            patient_info = self.cohort[self.cohort['subject_id'] == subject_id]
            
            if patient_info.empty:
                continue
                
            # Determine if this patient had an embolism
            has_embolism = patient_info['embolism_type'].max() > 0
            
            # Sort data by time
            subject_group = subject_group.sort_values('charttime')
            times = subject_group['charttime'].unique()
            
            if len(times) <= self.time_window:
                continue
                
            if has_embolism:
                # Create a positive window from the last time_window entries
                end_idx = len(times) - 1
                start_idx = max(0, end_idx - self.time_window + 1)
                
                window_data = subject_group[
                    (subject_group['charttime'] >= times[start_idx]) & 
                    (subject_group['charttime'] <= times[end_idx])
                ]
                
                if len(window_data) > 0:
                    windows.append({
                        'subject_id': subject_id,
                        'window_data': window_data,
                        'label': 1,  # positive case
                        'static_data': patient_info.iloc[0].to_dict()
                    })
            else:
                # For negative samples, create one random window
                # Random start point
                start_idx = np.random.randint(0, len(times) - self.time_window)
                end_idx = start_idx + self.time_window - 1
                
                window_data = subject_group[
                    (subject_group['charttime'] >= times[start_idx]) & 
                    (subject_group['charttime'] <= times[end_idx])
                ]
                
                if len(window_data) > 0:
                    windows.append({
                        'subject_id': subject_id,
                        'window_data': window_data,
                        'label': 0,  # negative case
                        'static_data': patient_info.iloc[0].to_dict()
                    })
        
        return windows
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get a sample"""
        window = self.windows[idx]
        
        # Extract time series data
        vital_columns = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temp', 'spo2']
        lab_columns = ['troponin_i', 'ddimer', 'platelets', 'inr', 'wbc']
        
        # Get all available time series columns
        ts_columns = [col for col in vital_columns + lab_columns if col in window['window_data'].columns]
        
        if not ts_columns:
            # Fallback if no time series data
            ts_data = np.zeros((1, len(vital_columns) + len(lab_columns)))
        else:
            ts_data = window['window_data'][ts_columns].values
            
            # Fill missing values with mean
            for col_idx in range(ts_data.shape[1]):
                col_data = ts_data[:, col_idx]
                mask = np.isnan(col_data)
                if mask.any():
                    col_mean = np.nanmean(col_data)
                    col_data[mask] = col_mean if not np.isnan(col_mean) else 0
                    ts_data[:, col_idx] = col_data
        
        # Get sequence length
        seq_length = len(ts_data)
        
        # Get static and note features
        static_features = []
        
        # Demographics
        if 'age' in window['static_data']:
            age = window['static_data']['age']
            static_features.append(age / 100.0)  # Normalize age
        else:
            static_features.append(0.5)  # Default

        if 'gender' in window['static_data']:
            gender = 1 if window['static_data']['gender'] == 'M' else 0
            static_features.append(gender)
        else:
            static_features.append(0.5)  # Default
            
        # Clinical note features
        note_columns = [
            'contains_shortness_of_breath', 'contains_chest_pain',
            'contains_leg_pain', 'contains_leg_swelling'
        ]
        
        for col in note_columns:
            if col in window['window_data'].columns:
                value = window['window_data'][col].max()
                static_features.append(float(value))
            else:
                static_features.append(0.0)
                
        # Medication features
        med_columns = [
            'med_heparin', 'med_enoxaparin', 'med_warfarin'
        ]
        
        for col in med_columns:
            if col in window['window_data'].columns:
                value = window['window_data'][col].max()
                static_features.append(float(value))
            else:
                static_features.append(0.0)
                
        # Convert to tensors
        ts_tensor = torch.FloatTensor(ts_data)
        static_tensor = torch.FloatTensor(static_features)
        label_tensor = torch.FloatTensor([window['label']])
        
        return {
            'ts_data': ts_tensor,
            'ts_length': seq_length,
            'static_data': static_tensor,
            'label': label_tensor,
            'subject_id': window['subject_id']
        }


def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: Batch of samples
        
    Returns:
        Collated batch
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x['ts_length'], reverse=True)
    
    # Get max sequence length
    max_len = max(sample['ts_length'] for sample in batch)
    
    # Pad time series data
    ts_data = []
    ts_lengths = []
    static_data = []
    labels = []
    subject_ids = []
    
    for sample in batch:
        # Pad time series
        ts = sample['ts_data']
        length = sample['ts_length']
        
        # Handle empty or single-feature time series
        if ts.dim() == 1:
            ts = ts.unsqueeze(1)  # Add feature dimension
            
        if length == 0:
            length = 1
            ts = torch.zeros((1, ts.size(1) if ts.dim() > 1 else 1))
        
        # Pad sequence
        if length < max_len:
            if ts.dim() == 1:
                padding = torch.zeros(max_len - length)
            else:
                padding = torch.zeros((max_len - length, ts.size(1)))
            padded_ts = torch.cat([ts, padding], dim=0)
        else:
            padded_ts = ts
        
        ts_data.append(padded_ts)
        ts_lengths.append(length)
        static_data.append(sample['static_data'])
        labels.append(sample['label'])
        subject_ids.append(sample['subject_id'])
    
    # Stack tensors
    ts_data = torch.stack(ts_data)
    ts_lengths = torch.tensor(ts_lengths)
    static_data = torch.stack(static_data)
    labels = torch.stack(labels)
    
    return {
        'ts_data': ts_data,
        'ts_lengths': ts_lengths,
        'static_data': static_data,
        'labels': labels,
        'subject_ids': subject_ids
    }