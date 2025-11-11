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

        # Default loss: MSE
        if criterion is None:
            criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for src, action_map, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)

                optimizer.zero_grad()
                try:
                    pred = self(src, action_map, decoder_inputs=src)
                except TypeError:
                    pred = self(src, action_map)

                # 🔹 Use the passed loss function
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

    def evaluate_model(self, test_dataset, device, batch_size=64, checkpoint_path=None, criterion = None):
        self.to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔹 Loading best checkpoint from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.eval()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for src, action_map, tgt in loader:
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