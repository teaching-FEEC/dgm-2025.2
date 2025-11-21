import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 

class BaseRopeModel(nn.Module):
    """Abstract base class for rope prediction models."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each child model must implement its own forward method.")

    def _compute_loss(self, criterion, pred, tgt, src, action):
        """Helper to call criterion with correct arguments robustly"""
        # 1. Try passing everything
        try:
            return criterion(pred, tgt, src, action)
        # Catch TypeError (wrong arg count) AND RuntimeError (boolean value of tensor)
        except (TypeError, RuntimeError):
            # 2. Try passing pred, tgt, src
            try:
                return criterion(pred, tgt, src)
            except (TypeError, RuntimeError):
                # 3. Fallback to standard loss
                return criterion(pred, tgt)

    def train_model(
        self,
        train_dataset,
        val_dataset,
        device,
        batch_size=256,
        epochs=10,
        lr=1e-3,
        checkpoint_path=None,
        criterion=None,
        decoder_inputs=None,
        early_stopping = False,
        patience=5,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        if criterion is None:
            criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            for src, action_map, tgt_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                tgt = tgt_seq 
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)

                optimizer.zero_grad()
                try:
                    pred = self(src, action_map, decoder_inputs=src)
                except TypeError:
                    pred = self(src, action_map)

                loss = self._compute_loss(criterion, pred, tgt, src, action_map)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            
            # 1. Validation (MUST come first)
            val_loss = self.evaluate_model(val_dataset, device, batch_size=batch_size, criterion=criterion)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

            # 2. Save best model
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Ensure directory exists
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.state_dict(), checkpoint_path)
                print(f"✅ Saved new best checkpoint at {checkpoint_path}")
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1 

            # 3. Early Stopping Check
            if epochs_no_improve >= patience and early_stopping:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break 

        return self

    def evaluate_model(self, test_dataset, device, batch_size=64, checkpoint_path=None, criterion=None):
        """
        Calculates loss on the data *as-is* (e.g., normalized loss).
        """
        self.to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔹 Loading best checkpoint from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.eval()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        
        if criterion is None:
            criterion = nn.MSELoss()

        with torch.no_grad():
            for src, action_map, tgt_seq in loader:
                tgt = tgt_seq 
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)
                try:
                    pred = self(src, action_map, decoder_inputs=src)
                except TypeError:
                    pred = self(src, action_map)

                loss = self._compute_loss(criterion, pred, tgt, src, action_map)

                total_loss += loss.item()
        return total_loss / len(loader)


    def evaluate_model_denormalized(
        self, 
        test_dataset, 
        device, 
        train_mean, 
        train_std, 
        batch_size=64, 
        checkpoint_path=None,
        criterion=None
    ):
        """
        Calculates loss in the *original, unnormalized* coordinate space.
        """
        self.to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔹 Loading best checkpoint from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.eval()
        
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)
        epsilon = 1e-8 

        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        
        if criterion is None:
            criterion = nn.MSELoss()

        with torch.no_grad():
            for src_norm, action_map, tgt_seq_norm in loader:
                tgt_norm = tgt_seq_norm 
                src_norm, action_map, tgt_norm = src_norm.to(device), action_map.to(device), tgt_norm.to(device)
                
                try:
                    pred_norm = self(src_norm, action_map, decoder_inputs=src_norm)
                except TypeError:
                    pred_norm = self(src_norm, action_map)

                pred_denorm = pred_norm * (train_std + epsilon) + train_mean
                tgt_denorm = tgt_norm * (train_std + epsilon) + train_mean
                src_denorm = src_norm * (train_std + epsilon) + train_mean

                loss = self._compute_loss(criterion, pred_denorm, tgt_denorm, src_denorm, action_map)

                total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate_autoregressive_rollout(
        self,
        test_src_tensor, 
        test_act_tensor, 
        test_tgt_tensor, 
        device,
        steps,
        criterion,
        checkpoint_path=None,
        denormalize_stats=None, 
        num_rollouts=100      
    ):
        """
        Calculates autoregressive rollout loss.
        """
        self.to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔹 Loading best checkpoint from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.eval()
        
        total_loss = 0.0
        
        required_samples = num_rollouts + steps - 1
        if len(test_src_tensor) < required_samples:
            num_rollouts_to_run = len(test_src_tensor) - steps + 1
            if num_rollouts_to_run <= 0:
                return 0.0
            print(f"Running {num_rollouts_to_run} rollouts instead.")
        else:
            num_rollouts_to_run = num_rollouts
            
        start_indices = range(num_rollouts_to_run)
        print(f"Running {num_rollouts_to_run} rollouts of {steps} steps each...")

        denorm = denormalize_stats is not None
        if denorm:
            train_mean, train_std = denormalize_stats
            train_mean = train_mean.to(device)
            train_std = train_std.to(device)
            epsilon = 1e-8

        with torch.no_grad():
            
            for i in tqdm(start_indices, desc=f"Autoregressive Rollout ({steps}-step)"):
                step_losses = []
                current_state_norm = test_src_tensor[i].unsqueeze(0).to(device)
                
                for k in range(steps):
                    current_action = test_act_tensor[i+k].unsqueeze(0).to(device)
                    try:
                        pred_next_state_norm = self(current_state_norm, current_action, decoder_inputs=current_state_norm)
                    except TypeError:
                        pred_next_state_norm = self(current_state_norm, current_action)

                    tgt_norm = test_tgt_tensor[i+k].unsqueeze(0).to(device)
                    
                    if denorm:
                        pred_denorm = pred_next_state_norm * (train_std + epsilon) + train_mean
                        tgt_denorm = tgt_norm * (train_std + epsilon) + train_mean
                        src_denorm = current_state_norm * (train_std + epsilon) + train_mean
                        
                        loss = self._compute_loss(criterion, pred_denorm, tgt_denorm, src_denorm, current_action)
                    else:
                        loss = self._compute_loss(criterion, pred_next_state_norm, tgt_norm, current_state_norm, current_action)
                    
                    step_losses.append(loss.item())
                    current_state_norm = pred_next_state_norm
                
                total_loss += np.mean(step_losses)

        return total_loss / num_rollouts_to_run