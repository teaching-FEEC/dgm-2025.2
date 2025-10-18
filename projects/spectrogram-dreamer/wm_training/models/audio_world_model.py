import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .encoders import MultiEncoder
from .rssm import RSSMCore
from .spectrogram_decoder import SpectrogramMultiDecoder
from .functions import *


class AudioWorldModel(nn.Module):
    
    def __init__(self, conf):
        super().__init__()
        
        self.conf = conf
        
        # Encoder
        self.encoder = MultiEncoder(conf)
        
        # RSSM Core
        self.rssm = RSSMCore(
            embed_dim=self.encoder.out_dim,
            action_dim=conf.action_dim,
            deter_dim=conf.deter_dim,
            stoch_dim=conf.stoch_dim,
            stoch_discrete=conf.stoch_discrete,
            hidden_dim=conf.hidden_dim,
            gru_layers=conf.gru_layers,
            gru_type=conf.gru_type,
            layer_norm=conf.layer_norm
        )
        
        # Decoder
        features_dim = conf.deter_dim + conf.stoch_dim * conf.stoch_discrete
        self.decoder = SpectrogramMultiDecoder(features_dim, conf)
        
        # KL balancing (DreamerV2 style)
        self.kl_balance = getattr(conf, 'kl_balance', 0.8)
        self.kl_free = getattr(conf, 'kl_free', 1.0)
        self.kl_weight = getattr(conf, 'kl_weight', 1.0)
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        reset: torch.Tensor,
        in_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        iwae_samples: int = 1,
        do_open_loop: bool = False
    ):
     
        T, B = obs['spectrogram_tokens'].shape[:2]
        
        # Inicializa estado se necessário
        if in_state is None:
            in_state = self.rssm.init_state(B * iwae_samples)
        
        # tokens -> embeddings
        embed = self.encoder(obs)  # (T, B, E)
        
        # embeddings -> latent states
        (priors, posts, samples, features, states, out_state) = self.rssm(
            embed=embed,
            action=actions,
            reset=reset,
            in_state=in_state,
            iwae_samples=iwae_samples,
            do_open_loop=do_open_loop
        )
        # priors: (T, B, I, 2S)
        # posts: (T, B, I, 2S)
        # features: (T, B, I, D+S)
        
        # features -> reconstrução
        loss_reconstr, decoder_metrics, decoder_tensors = self.decoder.training_step(
            features,
            obs,
            extra_metrics=True
        )
        
        # KL divergence
        prior_distr = self.rssm.zdistr(priors)
        post_distr = self.rssm.zdistr(posts)
        
        # KL com balanceamento
        kl_loss = torch.distributions.kl_divergence(post_distr, prior_distr)  # (T, B, I)
        kl_loss = torch.maximum(kl_loss, torch.tensor(self.kl_free).to(kl_loss.device))
        
        # Balanced KL (DreamerV2 Eq. 6)
        alpha = self.kl_balance
        kl_lhs = kl_loss.detach() * (1 - alpha) + kl_loss * alpha
        kl_rhs = kl_loss.detach() * alpha + kl_loss * (1 - alpha)
        kl_loss_balanced = kl_lhs + kl_rhs
        
        # Loss total
        # Reduz IWAE dimension
        loss_reconstr_tb = -logavgexp(-loss_reconstr, dim=2)  # (T, B, I) -> (T, B)
        kl_loss_tb = -logavgexp(-kl_loss_balanced, dim=2)  # (T, B, I) -> (T, B)
        
        loss_model = loss_reconstr_tb + self.kl_weight * kl_loss_tb  # (T, B)
        
        # Loss total (média sobre T e B)
        loss_total = loss_model.mean()
        
        # Métricas
        metrics = {
            'loss_total': loss_total.detach(),
            'loss_reconstr': loss_reconstr_tb.detach().mean(),
            'loss_kl': kl_loss_tb.detach().mean(),
            'kl_raw': kl_loss.detach().mean(),
            'prior_entropy': prior_distr.entropy().detach().mean(),
            'post_entropy': post_distr.entropy().detach().mean(),
            **decoder_metrics
        }
        
        # Tensors para análise
        tensors = {
            'loss_model': loss_model.detach(),
            'priors': priors.detach(),
            'posts': posts.detach(),
            'features': features.detach(),
            **decoder_tensors
        }
        
        return {
            'loss': loss_total,
            'metrics': metrics,
            'tensors': tensors,
            'out_state': out_state
        }
    
    def imagine(
        self,
        in_state: Tuple[torch.Tensor, torch.Tensor],
        actions: torch.Tensor,
        horizon: int
    ):
        if len(actions.shape) == 2:
            # Repete ação para todos os passos
            actions = actions.unsqueeze(0).expand(horizon, -1, -1)
        
        B = actions.shape[1]
        device = actions.device
        
        # Roll out com prior (sem observações)
        h, z = in_state
        states_h = []
        states_z = []
        
        for t in range(horizon):
            # RSSM forward com prior
            prior, (h, z) = self.rssm.cell.forward_prior(
                actions[t],
                reset_mask=None,
                in_state=(h, z)
            )
            states_h.append(h)
            states_z.append(z)
        
        # Concatena estados
        states_h = torch.stack(states_h)  # (H, B, D)
        states_z = torch.stack(states_z)  # (H, B, S)
        features = torch.cat([states_h, states_z], dim=-1)  # (H, B, D+S)
        
        # Decodifica (sem IWAE samples neste caso)
        features_expanded = features.unsqueeze(2)  # (H, B, 1, D+S)
        
        # Reconstrução
        token_logits = self.decoder.token_decoder(features_expanded)  # (H, B, 1, V)
        token_logits = token_logits.squeeze(2)  # (H, B, V)
        
        if self.decoder.token_decoder.mode == 'categorical':
            # Sample tokens
            token_probs = F.softmax(token_logits, dim=-1)
            token_indices = torch.argmax(token_probs, dim=-1)  # (H, B)
            
            return {
                'features': features,
                'token_logits': token_logits,
                'token_indices': token_indices,
                'token_probs': token_probs
            }
        else:
            # Continuous embeddings
            return {
                'features': features,
                'token_embeddings': token_logits
            }


@dataclass
class AudioWorldModelConfig:
    
    # Encoder config (herdado)
    spectrogram_tokens: bool = True
    spectrogram_encoder: str = 'token'
    spectrogram_vocab_size: int = 512
    spectrogram_embed_dim: int = 64
    spectrogram_out_dim: int = 256
    use_pretrained_embeddings: bool = False
    spectrogram_hidden_dim: int = 400
    spectrogram_hidden_layers: int = 2
    temporal_encoding: bool = True
    max_seq_length: int = 5000
    
    # RSSM config
    deter_dim: int = 512
    stoch_dim: int = 32
    stoch_discrete: int = 32
    hidden_dim: int = 512
    gru_layers: int = 1
    gru_type: str = 'gru'
    action_dim: int = 1
    
    # Decoder config
    decoder_mode: str = 'categorical'  # 'categorical' ou 'continuous'
    decoder_hidden_dim: int = 400
    decoder_hidden_layers: int = 2
    
    # Loss weights
    token_weight: float = 1.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8
    kl_free: float = 1.0
    
    # Optional
    predict_reward: bool = False
    predict_terminal: bool = False
    reward_weight: float = 0.1
    terminal_weight: float = 0.1
    
    # General
    layer_norm: bool = True
    
    # Disabled (para compatibilidade com MultiEncoder)
    image_encoder: Optional[str] = None
    image_channels: int = 0
    vecobs_size: int = 0
    reward_input: bool = False


def create_audio_world_model(
    vocab_size: int = 512,
    embed_dim: int = 64,
    deter_dim: int = 512,
    stoch_dim: int = 32,
    device: str = 'cpu'
) -> AudioWorldModel:
    
    config = AudioWorldModelConfig(
        spectrogram_vocab_size=vocab_size,
        spectrogram_embed_dim=embed_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim
    )
    
    model = AudioWorldModel(config)
    model = model.to(device)
    
    return model
