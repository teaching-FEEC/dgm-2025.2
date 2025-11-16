import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseRopeModel(nn.Module):
    """Abstract base class for rope prediction models."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each child model must implement its own forward method.")

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
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        if criterion is None:
            criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            # --- MODIFIED: Use tgt_seq from loader ---
            for src, action_map, tgt_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                # We only need t+1 for training
                tgt = tgt_seq # In train_ds, tgt_seq is just t+1
                
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)

                optimizer.zero_grad()
                try:
                    pred = self(src, action_map, decoder_inputs=src)
                except TypeError:
                    pred = self(src, action_map)

                loss = criterion(pred, tgt, src) if criterion.__code__.co_argcount > 2 else criterion(pred, tgt)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = self.evaluate_model(val_dataset, device, batch_size=batch_size, criterion=criterion)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

            # Save best model
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), checkpoint_path)
                print(f"Saved new best checkpoint at {checkpoint_path}")

        return self

    def evaluate_model(self, test_dataset, device, batch_size=64, criterion = None):
        """
        Calculates loss on the data *as-is* (e.g., normalized loss).
        """
        self.to(device)
        self.eval()
        
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            # --- MODIFIED: Use tgt_seq from loader ---
            for src, action_map, tgt_seq in loader:
                
                # We only need t+1 for this eval
                tgt = tgt_seq # In val_ds, tgt_seq is just t+1
                
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)
                try:
                    pred = self(src, action_map, decoder_inputs=src)
                except TypeError:
                    pred = self(src, action_map)

                if criterion is None:
                    loss = nn.functional.mse_loss(pred, tgt)
                else:
                    loss = criterion(pred, tgt, src) if criterion.__code__.co_argcount > 2 else criterion(pred, tgt)

                total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate_model_denormalized(
        self, 
        test_dataset, 
        device, 
        train_mean, 
        train_std, 
        batch_size=64, 
        criterion=None
    ):
        """
        Calculates loss in the *original, unnormalized* coordinate space.
        """
        self.to(device)
        self.eval()
        
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)
        epsilon = 1e-8 

        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        
        if criterion is None:
            criterion = nn.functional.mse_loss

        with torch.no_grad():
            # --- MODIFIED: Use tgt_seq from loader ---
            for src_norm, action_map, tgt_seq_norm in loader:
                
                # We only need t+1 for this eval
                tgt_norm = tgt_seq_norm # In val_ds, tgt_seq is just t+1
                
                src_norm, action_map, tgt_norm = src_norm.to(device), action_map.to(device), tgt_norm.to(device)
                
                try:
                    pred_norm = self(src_norm, action_map, decoder_inputs=src_norm)
                except TypeError:
                    pred_norm = self(src_norm, action_map)

                pred_denorm = pred_norm * (train_std + epsilon) + train_mean
                tgt_denorm = tgt_norm * (train_std + epsilon) + train_mean
                src_denorm = src_norm * (train_std + epsilon) + train_mean

                loss = criterion(pred_denorm, tgt_denorm, src_denorm) if criterion.__code__.co_argcount > 2 else criterion(pred_denorm, tgt_denorm)

                total_loss += loss.item()
        return total_loss / len(loader)

    # --- vvvv NEW FUNCTION vvvv ---
    def evaluate_autoregressive_rollout(
        self,
        test_src_tensor, # The full, normalized test src tensor
        test_act_tensor, # The full, raw test action tensor
        test_tgt_tensor, # The full, normalized test target tensor
        device,
        steps,
        criterion,
        denormalize_stats=None # (mean, std) if denorm loss is desired
    ):
        """
        Calculates autoregressive rollout loss for `steps` timesteps.
        Assumes Tensors are sequential and NOT in a DataLoader.
        """
        self.to(device)
        self.eval()
        
        total_loss = 0.0
        num_samples = len(test_src_tensor) - steps # We can't roll out the last few samples
        
        # Prepare denorm stats if provided
        denorm = denormalize_stats is not None
        if denorm:
            train_mean, train_std = denormalize_stats
            train_mean = train_mean.to(device)
            train_std = train_std.to(device)
            epsilon = 1e-8

        with torch.no_grad():
            # We must iterate sample by sample for a true rollout
            for i in tqdm(range(num_samples), desc="Autoregressive Rollout"):
                step_losses = []
                
                # Get the first state
                current_state_norm = test_src_tensor[i].unsqueeze(0).to(device)
                
                for k in range(steps):
                    # Get the action for this step
                    current_action = test_act_tensor[i+k].unsqueeze(0).to(device)
                    
                    # Predict the next state
                    try:
                        pred_next_state_norm = self(current_state_norm, current_action, decoder_inputs=current_state_norm)
                    except TypeError:
                        pred_next_state_norm = self(current_state_norm, current_action)

                    # Get the ground truth for this step
                    # The target for src[i] at step k is tgt[i+k]
                    tgt_norm = test_tgt_tensor[i+k].unsqueeze(0).to(device)
                    
                    # --- Calculate Loss ---
                    if denorm:
                        # Denormalize both pred and target before loss
                        pred_denorm = pred_next_state_norm * (train_std + epsilon) + train_mean
                        tgt_denorm = tgt_norm * (train_std + epsilon) + train_mean
                        loss = criterion(pred_denorm, tgt_denorm)
                    else:
                        # Calculate loss on normalized data
                        loss = criterion(pred_next_state_norm, tgt_norm)
                    
                    step_losses.append(loss.item())
                    
                    # The prediction becomes the next input
                    current_state_norm = pred_next_state_norm
                
                # Average the loss over the 10 steps for this one sample
                total_loss += np.mean(step_losses)

        # Return the average loss over all samples
        return total_loss / num_samples
    # --- ^^^^ NEW FUNCTION ^^^^ ---