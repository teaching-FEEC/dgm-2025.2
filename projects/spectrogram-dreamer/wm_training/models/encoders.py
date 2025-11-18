from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


class MultiEncoder(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.reward_input = conf.reward_input
        
        # Encoder para imagens (para ambientes visuais tradicionais)
        if hasattr(conf, 'image_channels') and conf.image_channels:
            if conf.reward_input:
                encoder_channels = conf.image_channels + 2  # + reward, terminal
            else:
                encoder_channels = conf.image_channels

            if conf.image_encoder == 'cnn':
                self.encoder_image = ConvEncoder(in_channels=encoder_channels,
                                                 cnn_depth=conf.cnn_depth)
            elif conf.image_encoder == 'dense':
                self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
                                                  out_dim=256,
                                                  hidden_layers=conf.image_encoder_layers,
                                                  layer_norm=conf.layer_norm)
            elif not conf.image_encoder:
                self.encoder_image = None
            else:
                assert False, conf.image_encoder
        else:
            self.encoder_image = None

        # Encoder para tokens de espectrogramas de áudio
        if hasattr(conf, 'spectrogram_tokens') and conf.spectrogram_tokens:
            encoder_type = getattr(conf, 'spectrogram_encoder', 'token')  # 'token' ou 'conv1d'
            
            if encoder_type == 'token':
                self.encoder_spectrogram = SpectrogramTokenEncoder(
                    num_embeddings=getattr(conf, 'spectrogram_vocab_size', 512),
                    embedding_dim=getattr(conf, 'spectrogram_embed_dim', 64),
                    out_dim=getattr(conf, 'spectrogram_out_dim', 256),
                    use_pretrained_embeddings=getattr(conf, 'use_pretrained_embeddings', False),
                    hidden_dim=getattr(conf, 'spectrogram_hidden_dim', 400),
                    hidden_layers=getattr(conf, 'spectrogram_hidden_layers', 2),
                    layer_norm=conf.layer_norm,
                    temporal_encoding=getattr(conf, 'temporal_encoding', True),
                    max_seq_length=getattr(conf, 'max_seq_length', 5000)
                )
            elif encoder_type == 'conv1d':
                self.encoder_spectrogram = Spectrogram1DConvEncoder(
                    num_embeddings=getattr(conf, 'spectrogram_vocab_size', 512),
                    embedding_dim=getattr(conf, 'spectrogram_embed_dim', 64),
                    out_dim=getattr(conf, 'spectrogram_out_dim', 256),
                    use_pretrained_embeddings=getattr(conf, 'use_pretrained_embeddings', False),
                    conv_channels=getattr(conf, 'conv_channels', [128, 256, 256]),
                    kernel_sizes=getattr(conf, 'kernel_sizes', [5, 5, 3]),
                    layer_norm=conf.layer_norm
                )
            else:
                assert False, f"Unknown spectrogram encoder type: {encoder_type}"
        else:
            self.encoder_spectrogram = None

        # Encoder para observações vetoriais
        if hasattr(conf, 'vecobs_size') and conf.vecobs_size:
            self.encoder_vecobs = MLP(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None

        assert self.encoder_image or self.encoder_vecobs or self.encoder_spectrogram, \
            "Either image_encoder, vecobs_size, or spectrogram_tokens should be set"
        
        self.out_dim = (
            (self.encoder_image.out_dim if self.encoder_image else 0) +
            (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0) +
            (self.encoder_spectrogram.out_dim if self.encoder_spectrogram else 0)
        )

    def forward(self, obs: Dict[str, Tensor]) -> TensorTBE:
        # TODO:
        #  1) Make this more generic, e.g. working without image input or without vecobs
        #  2) Treat all inputs equally, adding everything via linear layer to embed_dim

        embeds = []

        if self.encoder_image and 'image' in obs:
            image = obs['image']
            T, B, C, H, W = image.shape
            if self.reward_input:
                reward = obs['reward']
                terminal = obs['terminal']
                reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                image = torch.cat([image,  # (T,B,C+2,H,W)
                                reward_plane.to(image.dtype),
                                terminal_plane.to(image.dtype)], dim=-3)

            embed_image = self.encoder_image.forward(image)  # (T,B,E)
            embeds.append(embed_image)

        if self.encoder_spectrogram and 'spectrogram_tokens' in obs:
            # obs['spectrogram_tokens'] pode ser:
            # - (T, B) para índices discretos
            # - (T, B, E) para embeddings contínuos pré-computados
            embed_spectrogram = self.encoder_spectrogram.forward(obs['spectrogram_tokens'])  # (T,B,E)
            embeds.append(embed_spectrogram)

        if self.encoder_vecobs and 'vecobs' in obs:
            embed_vecobs = self.encoder_vecobs(obs['vecobs'])
            embeds.append(embed_vecobs)

        embed = torch.cat(embeds, dim=-1)  # (T,B,E+...)
        return embed


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            activation(),
            nn.Flatten()
        )

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class SpectrogramTokenEncoder(nn.Module):

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        out_dim: int = 256,
        use_pretrained_embeddings: bool = False,
        hidden_dim: int = 400,
        hidden_layers: int = 2,
        layer_norm: bool = True,
        activation=nn.ELU,
        temporal_encoding: bool = True,
        max_seq_length: int = 5000
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.temporal_encoding = temporal_encoding
        
        # Embedding layer (apenas se não usar embeddings pré-treinados)
        if not use_pretrained_embeddings:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            # Inicialização Xavier/Glorot
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            self.embedding = None
        
        # Positional encoding temporal (opcional)
        if temporal_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(max_seq_length, 1, embedding_dim) * 0.02
            )
        
        # MLP para projetar embeddings para o espaço do RSSM
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        
        # Primeira camada
        layers += [
            nn.Linear(embedding_dim, hidden_dim),
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
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()
        ]
        
        self.projection = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> TensorTBE:
        """
        Args:
            x: Tokens de entrada
               - Se use_pretrained_embeddings=False: IntTensor de shape (T, B) com índices
               - Se use_pretrained_embeddings=True: FloatTensor de shape (T, B, embedding_dim)
        
        Returns:
            Tensor (T, B, out_dim): Features para o RSSM
        """
        
        # Converte índices para embeddings se necessário
        if not self.use_pretrained_embeddings:
            if x.dtype not in [torch.long, torch.int32, torch.int64]:
                x = x.long()
            x = self.embedding(x)  # (T, B) -> (T, B, embedding_dim)
        
        # Adiciona positional encoding se habilitado
        if self.temporal_encoding:
            T = x.shape[0]
            x = x + self.positional_encoding[:T, :, :]  # (T, B, embedding_dim)
        
        # Projeta para o espaço de features do RSSM
        # flatten_batch para processar (T*B, embedding_dim)
        x, bd = flatten_batch(x, 1)  # (T, B, E) -> (T*B, E)
        y = self.projection(x)  # (T*B, out_dim)
        y = unflatten_batch(y, bd)  # (T*B, out_dim) -> (T, B, out_dim)
        
        return y


class Spectrogram1DConvEncoder(nn.Module):
    """
    Encoder alternativo que usa Conv1D para processar sequências de tokens
    de espectrogramas, similar a processamento de texto com CNNs.
    Útil para capturar padrões locais temporais nos espectrogramas.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        out_dim: int = 256,
        use_pretrained_embeddings: bool = False,
        conv_channels: list = [128, 256, 256],
        kernel_sizes: list = [5, 5, 3],
        layer_norm: bool = True,
        activation=nn.ELU
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        
        # Embedding layer
        if not use_pretrained_embeddings:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            self.embedding = None
        
        # Conv1D layers
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        
        in_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2  # Same padding
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                activation()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Camada de projeção final
        self.projection = nn.Sequential(
            nn.Linear(conv_channels[-1], out_dim),
            norm(out_dim, eps=1e-3) if layer_norm else nn.Identity(),
            activation()
        )

    def forward(self, x: Tensor) -> TensorTBE:
        """
        Args:
            x: Tokens de entrada (T, B) ou (T, B, embedding_dim)
        
        Returns:
            Tensor (T, B, out_dim): Features para o RSSM
        """
        T, B = x.shape[:2]
        
        # Converte índices para embeddings se necessário
        if not self.use_pretrained_embeddings:
            if x.dtype not in [torch.long, torch.int32, torch.int64]:
                x = x.long()
            x = self.embedding(x)  # (T, B) -> (T, B, embedding_dim)
        
        # Reshape para Conv1d: (T, B, E) -> (B, E, T)
        x = x.permute(1, 2, 0)
        
        # Aplica convoluções
        x = self.conv_layers(x)  # (B, C, T)
        
        # Reshape de volta: (B, C, T) -> (T, B, C)
        x = x.permute(2, 0, 1)
        
        # Projeção final
        x, bd = flatten_batch(x, 1)
        y = self.projection(x)
        y = unflatten_batch(y, bd)
        
        return y
