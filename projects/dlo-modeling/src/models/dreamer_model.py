import torch
import inspect
import torch.nn.functional as F
import torch.optim as optim
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import Normal, kl
from base_model import BaseRopeModel 
from tqdm import tqdm

class Encoder(nn.Module):
    """
    Encodes the rope state S_t (B, L, 3) into an embedding e_t (B, d_embed).
    Uses 1D Convolutions to process the sequence.
    """
    def __init__(self, L=70, d_embed=256, channels=64):
        super().__init__()
        self.d_embed = d_embed

        # We permute input to (B, C, L) = (B, 3, L)
        self.conv_net = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=5, stride=2, padding=2), # (B, C, L/2)
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 2 * channels, kernel_size=5, stride=2, padding=2), # (B, 2C, L/4)
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * channels, 4 * channels, kernel_size=5, stride=2, padding=2), # (B, 4C, L/8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), # (B, 4C, 1)
            nn.Flatten() # (B, 4C)
        )

        self.fc = nn.Linear(4 * channels, d_embed)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Input state: (B, L, 3)
        # Conv1d expects (B, C, L)
        x = state.permute(0, 2, 1)
        x = self.conv_net(x)
        e = self.fc(x)
        return e

class ActionEncoder(nn.Module):
    """
    Encodes the action A_t (B, 4) [x, y, z, link_id] into an embedding a_t (B, d_action).
    """
    def __init__(self, num_links=70, d_link_embed=32, d_action_out=64):
        super().__init__()
        self.link_embed = nn.Embedding(num_links, d_link_embed)

        # MLP for continuous [x, y, z] part
        self.xyz_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_link_embed) # Project to same dim as link_embed
        )

        # Final MLP to combine
        self.final_mlp = nn.Sequential(
            nn.Linear(d_link_embed * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_action_out)
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # action: (B, 4)
        xyz = action[:, :3]
        link_id = action[:, 3].long()

        xyz_emb = self.xyz_mlp(xyz)        # (B, d_link_embed)
        link_emb = self.link_embed(link_id) # (B, d_link_embed)

        combined = torch.cat([xyz_emb, link_emb], dim=1) # (B, d_link_embed * 2)
        a = self.final_mlp(combined)
        return a

class Decoder(nn.Module):
    """
    Reconstructs the rope state S_hat_t (B, L, 3)
    from the latent state (h_t, z_t) -> (B, d_rnn + d_z).

    Uses a simple but effective MLP expansion.
    """
    def __init__(self, d_rnn=512, d_z=32, L=70, d_model=256):
        super().__init__()
        self.L = L
        self.d_model = d_model

        # Expand the single latent vector to a sequence
        self.fc_expand = nn.Linear(d_rnn + d_z, L * d_model)

        # Point-wise MLP to decode each sequence element
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3) # Project to [x, y, z]
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: (B, d_rnn), z: (B, d_z)
        latent_state = torch.cat([h, z], dim=1) # (B, d_rnn + d_z)

        x = self.fc_expand(latent_state) # (B, L * d_model)
        x = x.view(-1, self.L, self.d_model) # (B, L, d_model)

        s_hat = self.mlp(x) # (B, L, 3)
        return s_hat

class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM).
    Manages the latent state transition.

    (h_t, c_t) = f((h_{t-1}, c_{t-1}), z_{t-1}, a_{t-1})
    z_t ~ q(z_t | h_t, e_t)   (Posterior, for training)
    z_t ~ p(z_t | h_t)       (Prior, for dreaming)
    """
    def __init__(self, d_action=64, d_rnn=512, d_z=32, d_embed=256):
        super().__init__()
        self.d_rnn = d_rnn
        self.d_z = d_z

        # Recurrent cell
        self.lstm_cell = nn.LSTMCell(d_z + d_action, d_rnn)

        # Posterior: q(z_t | h_t, e_t)
        self.fc_posterior = nn.Sequential(
            nn.Linear(d_rnn + d_embed, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * d_z) # mu and std
        )

        # Prior: p(z_t | h_t)
        self.fc_prior = nn.Sequential(
            nn.Linear(d_rnn, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * d_z) # mu and std
        )

    def _get_dist(self, fc_output: torch.Tensor) -> Normal:
        """Helper to create a Normal distribution from FC output."""
        mu, std = fc_output.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1 # Ensure std is positive and non-zero
        return Normal(mu, std)

    def observe(self, h_prev: torch.Tensor, c_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor, e_t: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, Normal, Normal):
        """
        Calculates the posterior latent state z_t used for training.
        """
        # 1. Update recurrent state
        rnn_input = torch.cat([z_prev, a_prev], dim=-1)
        (h_t, c_t) = self.lstm_cell(rnn_input, (h_prev, c_prev))

        # 2. Get Posterior (using current observation e_t)
        post_input = torch.cat([h_t, e_t], dim=-1)
        post_dist = self._get_dist(self.fc_posterior(post_input))

        # 3. Get Prior (for loss calculation)
        prior_dist = self._get_dist(self.fc_prior(h_t))

        # 4. Sample z_t from posterior for next step
        z_t = post_dist.rsample() # Use reparameterization trick

        return h_t, c_t, z_t, post_dist, prior_dist

    def dream_step(self, h_prev: torch.Tensor, c_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Predicts the next latent state (h_t, z_t) using the prior.
        Used for "dreaming" (inference/prediction).
        """
        # 1. Update recurrent state
        rnn_input = torch.cat([z_prev, a_prev], dim=-1)
        (h_t, c_t) = self.lstm_cell(rnn_input, (h_prev, c_prev))

        # 2. Get Prior
        prior_dist = self._get_dist(self.fc_prior(h_t))

        # 3. Sample z_t from prior
        z_t = prior_dist.sample() # No reparam trick needed for inference

        return h_t, c_t, z_t

# --- Refactored DreamerRopeModel ---

class DreamerRopeModel(BaseRopeModel):
    """
    The complete Dreamer-like World Model.
    Combines all components.
    """
    def __init__(self, L=70, d_embed=256, d_action=64, d_rnn=512, d_z=32, beta_kl=1.0, recon_loss_fn=None):
        super().__init__()
        self.L = L
        self.d_rnn = d_rnn
        self.d_z = d_z
        self.d_action = d_action
        self.beta_kl = beta_kl # Store KL loss weight

        if recon_loss_fn is None:
            # Default to standard MSE if nothing provided
            self.recon_loss_fn = nn.MSELoss()
        else:
            if not isinstance(recon_loss_fn, nn.Module):
                raise TypeError("recon_loss_fn must be an instance of torch.nn.Module")
            self.recon_loss_fn = recon_loss_fn

        self.encoder = Encoder(L=L, d_embed=d_embed)
        self.action_encoder = ActionEncoder(num_links=L, d_action_out=d_action)
        self.rssm = RSSM(d_action=d_action, d_rnn=d_rnn, d_z=d_z, d_embed=d_embed)
        self.decoder = Decoder(d_rnn=d_rnn, d_z=d_z, L=L)

    def get_initial_hidden_state(self, batch_size: int, device: str = 'cpu') -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Returns zero tensors for the initial (h_0, c_0, z_0)."""
        return (
            torch.zeros(batch_size, self.d_rnn, device=device),
            torch.zeros(batch_size, self.d_rnn, device=device),
            torch.zeros(batch_size, self.d_z, device=device)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, Normal, Normal):
        """
        The training forward pass.
        Unrolls the model over a sequence of length T.
        """
        B, T, _, _ = states.shape
        device = states.device

        # Initialize
        h_prev, c_prev, z_prev = self.get_initial_hidden_state(B, device=device)
        # Create a dummy "A_{-1}"
        a_prev = self.action_encoder(torch.zeros(B, 4, device=device))

        # Lists to store outputs
        h_states, c_states, z_states, post_dists_mu, post_dists_std, prior_dists_mu, prior_dists_std = [], [], [], [], [], [], []

        for t in range(T):
            # 1. Get current observation and action
            S_t = states[:, t]
            A_t = actions[:, t]

            # 2. Encode observation
            e_t = self.encoder(S_t)

            # 3. Update RSSM state using posterior
            h_t, c_t, z_t, post_t, prior_t = self.rssm.observe(h_prev, c_prev, z_prev, a_prev, e_t)

            # 4. Store states and distributions
            h_states.append(h_t)
            c_states.append(c_t) # Store cell state
            z_states.append(z_t)
            post_dists_mu.append(post_t.mean)
            post_dists_std.append(post_t.stddev)
            prior_dists_mu.append(prior_t.mean)
            prior_dists_std.append(prior_t.stddev)

            # 5. Update prev states for next loop
            h_prev, c_prev, z_prev = h_t, c_t, z_t
            a_prev = self.action_encoder(A_t) # a_prev for next step is A_t

        # Stack all lists into tensors
        h_T = torch.stack(h_states, dim=1)     # (B, T, d_rnn)
        z_T = torch.stack(z_states, dim=1)     # (B, T, d_z)

        # 6. Reconstruct all states in parallel from latents
        # (Decoder only needs h and z, not c)
        recon_states = self.decoder(
            h_T.reshape(B * T, self.d_rnn),
            z_T.reshape(B * T, self.d_z)
        ).view(B, T, self.L, 3)

        # Recreate distribution objects for loss calculation
        post_dists = Normal(torch.stack(post_dists_mu, dim=1), torch.stack(post_dists_std, dim=1))
        prior_dists = Normal(torch.stack(prior_dists_mu, dim=1), torch.stack(prior_dists_std, dim=1))

        return recon_states, post_dists, prior_dists

    def observe(self, S_t: torch.Tensor, A_t_minus_1: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, z_prev: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, Normal, Normal):
        """
        Grounds the model in reality by observing a single state S_t.
        """
        is_batched = S_t.dim() == 3
        if not is_batched:
            S_t = S_t.unsqueeze(0)
            A_t_minus_1 = A_t_minus_1.unsqueeze(0)
            h_prev = h_prev.unsqueeze(0)
            c_prev = c_prev.unsqueeze(0)
            z_prev = z_prev.unsqueeze(0)

        e_t = self.encoder(S_t)
        a_prev_emb = self.action_encoder(A_t_minus_1)
        h_t, c_t, z_t, post_t, prior_t = self.rssm.observe(h_prev, c_prev, z_prev, a_prev_emb, e_t)

        if not is_batched:
             h_t, c_t, z_t = h_t.squeeze(0), c_t.squeeze(0), z_t.squeeze(0)

        return h_t, c_t, z_t, post_t, prior_t

    def dream(self, h_t: torch.Tensor, c_t: torch.Tensor, z_t: torch.Tensor, A_t: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Dreams one step into the future.
        """
        is_batched = h_t.dim() == 2
        if not is_batched:
            h_t, c_t, z_t, A_t = h_t.unsqueeze(0), c_t.unsqueeze(0), z_t.unsqueeze(0), A_t.unsqueeze(0)

        a_t_emb = self.action_encoder(A_t)

        # Predict next latent state using prior
        h_tp1, c_tp1, z_tp1 = self.rssm.dream_step(h_t, c_t, z_t, a_t_emb)

        # Decode the dream
        S_hat_tp1 = self.decoder(h_tp1, z_tp1)

        if not is_batched:
            S_hat_tp1, h_tp1, c_tp1, z_tp1 = S_hat_tp1.squeeze(0), h_tp1.squeeze(0), c_tp1.squeeze(0), z_tp1.squeeze(0)

        return S_hat_tp1, h_tp1, c_tp1, z_tp1

    # --- Overridden Training and Evaluation Methods ---

    def calculate_loss(
        self,
        states: torch.Tensor,
        recon_states: torch.Tensor,
        post_dists: Normal,
        prior_dists: Normal,
        recon_criterion = None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculates the total loss for the Dreamer model.
        Loss = Reconstruction Loss + beta * KL Loss
        """

        # 1. Reconstruction Loss
        # Define tensors (slice time to match dynamics if needed)
        pred = recon_states[:, 1:] # S_hat_1 ... S_hat_{T-1} (Shape T-1)
        tgt = states[:, 1:]        # S_1     ... S_{T-1}     (Shape T-1)
        src = states[:, :-1]       # S_0     ... S_{T-2}     (Shape T-1)

        # Determine which loss function to use
        loss_fn = recon_criterion if recon_criterion is not None else self.recon_loss_fn

        # Robust argument checking for nn.Module
        use_three_args = False
        try:
            # Since we enforced nn.Module, we can inspect the forward method
            if hasattr(loss_fn, 'forward'):
                sig = inspect.signature(loss_fn.forward)
                # 'self' is implicit in bound methods inspection, so we just count params
                # For PhysicsInformedRopeLoss.forward(self, pred, tgt, src), params are pred, tgt, src.
                if len(sig.parameters) >= 3:
                    use_three_args = True
        except (ValueError, TypeError):
            pass

        if use_three_args:
             # Use 3-arg loss: (pred, tgt, src)
             recon_loss = loss_fn(pred, tgt, src)
        else:
             # Use 2-arg loss: (pred, tgt)
             # Note: We use the sliced tensors (pred, tgt) for consistency
             recon_loss = loss_fn(pred, tgt)

        # 2. KL Divergence Loss
        post_dist_detached = Normal(post_dists.mean.detach(), post_dists.stddev.detach())
        prior_dist_detached = Normal(prior_dists.mean, prior_dists.stddev.detach()) # Detach prior stddev only

        kl_loss_detached = kl.kl_divergence(post_dist_detached, prior_dists)
        kl_loss_detached = torch.sum(kl_loss_detached, dim=-1) # Sum over d_z
        kl_loss_detached = torch.mean(kl_loss_detached) # Mean over B and T

        kl_loss_post = kl.kl_divergence(post_dists, prior_dist_detached)
        kl_loss_post = torch.sum(kl_loss_post, dim=-1)
        kl_loss_post = torch.mean(kl_loss_post)

        kl_loss = 0.8 * kl_loss_detached + 0.2 * kl_loss_post

        # 3. Total Loss
        total_loss = recon_loss + self.beta_kl * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_model(
        self,
        train_dataset,
        val_dataset,
        device,
        batch_size=32,
        epochs=10,
        lr=1e-4,
        checkpoint_path=None,
        recon_loss_fn=None,
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Check if passed recon_loss_fn is valid
        if recon_loss_fn is not None and not isinstance(recon_loss_fn, nn.Module):
             raise TypeError("recon_loss_fn passed to train_model must be an instance of torch.nn.Module")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.train()
            train_loss, train_recon, train_kl = 0.0, 0.0, 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for states, actions in pbar:
                states, actions = states.to(device), actions.to(device)

                optimizer.zero_grad()

                recon_states, post_dists, prior_dists = self(states, actions)

                # Pass the specific loss function for this training run
                loss, recon, kl = self.calculate_loss(
                    states,
                    recon_states,
                    post_dists,
                    prior_dists,
                    recon_criterion=recon_loss_fn
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
                optimizer.step()

                train_loss += loss.item()
                train_recon += recon.item()
                train_kl += kl.item()
                pbar.set_postfix({
                    "loss": train_loss / (pbar.n + 1),
                    "recon": train_recon / (pbar.n + 1),
                    "kl": train_kl / (pbar.n + 1)
                })

            train_loss /= len(train_loader)
            train_recon /= len(train_loader)
            train_kl /= len(train_loader)

            # Validation
            val_loss, val_recon, val_kl = self.evaluate_model(val_dataset, device, batch_size=batch_size, criterion=recon_loss_fn)
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss={train_loss:.6f} | Recon={train_recon:.6f} | KL={train_kl:.6f}")
            print(f"  Val   Loss={val_loss:.6f} | Recon={val_recon:.6f} | KL={val_kl:.6f}")

            # Save best model
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), checkpoint_path)
                print(f"Saved new best checkpoint at {checkpoint_path}")

        return self

    def evaluate_model(self, test_dataset, device, batch_size=32, checkpoint_path=None, criterion=None):
        """
        OVERRIDDEN evaluation loop for the sequence-based Dreamer model.
        """
        self.to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading best checkpoint from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Check if passed criterion is valid
        if criterion is not None and not isinstance(criterion, nn.Module):
             raise TypeError("criterion passed to evaluate_model must be an instance of torch.nn.Module")

        self.eval()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        with torch.no_grad():
            for states, actions in loader:
                states, actions = states.to(device), actions.to(device)

                recon_states, post_dists, prior_dists = self(states, actions)

                loss, recon, kl = self.calculate_loss(states, recon_states, post_dists, prior_dists, recon_criterion=criterion)

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        n_batches = len(loader)
        if n_batches == 0:
             print("Warning: evaluate_model called with empty dataset.")
             return 0.0, 0.0, 0.0
        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches