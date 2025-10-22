import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .functions import *
from .common import *


class SpectrogramTokenDecoder(nn.Module):

    def __init__(
        self,
        in_dim: int,                    # Feature dim do RSSM (deter_dim + stoch_dim)
        vocab_size: int = 512,          # Tamanho do vocabulário
        embedding_dim: int = 64,        # Dimensão dos embeddings VQ-VAE
        hidden_dim: int = 400,
        hidden_layers: int = 2,
        layer_norm: bool = True,
        mode: str = 'categorical',      # 'categorical' ou 'continuous'
        activation=nn.ELU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mode = mode
        
        # Determina output dimension
        if mode == 'categorical':
            out_dim = vocab_size  # Logits sobre vocabulário
        elif mode == 'continuous':
            out_dim = embedding_dim  # Embeddings contínuos
        else:
            raise ValueError(f"Mode must be 'categorical' or 'continuous', got {mode}")
        
        # MLP decoder
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        
        # Primeira camada
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()
        ]
        
        # Camadas ocultas
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
        
        # Camada de saída
        layers += [nn.Linear(hidden_dim, out_dim)]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, features: Tensor) -> Tensor:

        features, bd = flatten_batch(features)
        output = self.model(features)
        output = unflatten_batch(output, bd)
        return output
    
    def loss_categorical(self, logits: Tensor, target_indices: Tensor) -> Tensor:

        T, B, I, V = logits.shape
        
        # Expande target para incluir dimensão I
        target = insert_dim(target_indices, 2, I)  # (T, B) -> (T, B, I)
        
        # Flatten para cross entropy
        logits_flat = logits.reshape(T * B * I, V)
        target_flat = target.reshape(T * B * I)
        
        # Cross entropy
        loss_flat = F.cross_entropy(logits_flat, target_flat, reduction='none')
        loss = loss_flat.reshape(T, B, I)
        
        return loss
    
    def loss_continuous(self, pred_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:

        T, B, I, E = pred_embeddings.shape
        
        # Expande target
        target = insert_dim(target_embeddings, 2, I)  # (T, B, E) -> (T, B, I, E)
        
        # MSE
        loss = 0.5 * torch.square(pred_embeddings - target).sum(dim=-1)  # (T, B, I)
        
        return loss
    
    def training_step(
        self,
        features: Tensor,      # (T, B, I, F)
        target: Tensor         # (T, B) ou (T, B, E)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        assert len(features.shape) == 4, f"Features deve ter shape (T,B,I,F), got {features.shape}"
        
        # Forward
        output = self.forward(features)  # (T, B, I, V) ou (T, B, I, E)
        
        # Loss
        if self.mode == 'categorical':
            assert len(target.shape) == 2, "Target deve ser (T, B) para modo categorical"
            loss_tbi = self.loss_categorical(output, target)
        else:  # continuous
            assert len(target.shape) == 3, "Target deve ser (T, B, E) para modo continuous"
            loss_tbi = self.loss_continuous(output, target)
        
        # Reduz IWAE dimension
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # (T, B, I) -> (T, B)
        
        # Reconstrução média
        reconstruction = output.mean(dim=2)  # (T, B, I, ...) -> (T, B, ...)
        
        return loss_tbi, loss_tb, reconstruction


class SpectrogramMultiDecoder(nn.Module):
    
    def __init__(self, features_dim, conf):
        super().__init__()
        
        # Pesos das losses
        self.token_weight = getattr(conf, 'token_weight', 1.0)
        self.reward_weight = getattr(conf, 'reward_weight', 0.1)
        self.terminal_weight = getattr(conf, 'terminal_weight', 0.1)
        
        # Decoder principal de tokens
        self.token_decoder = SpectrogramTokenDecoder(
            in_dim=features_dim,
            vocab_size=getattr(conf, 'spectrogram_vocab_size', 512),
            embedding_dim=getattr(conf, 'spectrogram_embed_dim', 64),
            hidden_dim=getattr(conf, 'decoder_hidden_dim', 400),
            hidden_layers=getattr(conf, 'decoder_hidden_layers', 2),
            layer_norm=conf.layer_norm,
            mode=getattr(conf, 'decoder_mode', 'categorical')
        )
        
        # Decoders opcionais para reward/terminal
        if getattr(conf, 'predict_reward', False):
            from .decoders import DenseNormalDecoder
            self.reward = DenseNormalDecoder(
                in_dim=features_dim,
                hidden_layers=2,
                layer_norm=conf.layer_norm
            )
        else:
            self.reward = None
        
        if getattr(conf, 'predict_terminal', False):
            from .decoders import DenseBernoulliDecoder
            self.terminal = DenseBernoulliDecoder(
                in_dim=features_dim,
                hidden_layers=2,
                layer_norm=conf.layer_norm
            )
        else:
            self.terminal = None
    
    def training_step(
        self,
        features: Tensor,           # (T, B, I, F)
        obs: dict,
        extra_metrics: bool = False
    ) -> Tuple[Tensor, dict, dict]:
        
        tensors = {}
        metrics = {}
        loss_reconstr = 0
        
        # Token reconstruction
        loss_token_tbi, loss_token, token_rec = self.token_decoder.training_step(
            features,
            obs['spectrogram_tokens']
        )
        loss_reconstr = loss_reconstr + self.token_weight * loss_token_tbi
        metrics['loss_token'] = loss_token.detach().mean()
        tensors['loss_token'] = loss_token.detach()
        tensors['token_rec'] = token_rec.detach()
        
        # Reward (opcional)
        if self.reward is not None and 'reward' in obs:
            loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(
                features,
                obs['reward']
            )
            loss_reconstr = loss_reconstr + self.reward_weight * loss_reward_tbi
            metrics['loss_reward'] = loss_reward.detach().mean()
            tensors['loss_reward'] = loss_reward.detach()
            tensors['reward_rec'] = reward_rec.detach()
        
        # Terminal (opcional)
        if self.terminal is not None and 'terminal' in obs:
            loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(
                features,
                obs['terminal']
            )
            loss_reconstr = loss_reconstr + self.terminal_weight * loss_terminal_tbi
            metrics['loss_terminal'] = loss_terminal.detach().mean()
            tensors['loss_terminal'] = loss_terminal.detach()
            tensors['terminal_rec'] = terminal_rec.detach()
        
        return loss_reconstr, metrics, tensors
