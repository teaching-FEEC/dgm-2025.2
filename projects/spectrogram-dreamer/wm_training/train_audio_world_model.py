import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_training.models.audio_world_model import AudioWorldModel, AudioWorldModelConfig
from wm_training.audio_dataset import create_dataloader, create_audio_dataloader

class AudioWorldModelTrainer:
    def __init__(
        self,
        model: AudioWorldModel,
        config: AudioWorldModelConfig,
        device: str = 'cuda',
        log_dir: str = 'runs/audio_world_model',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer (DreamerV2 usa Adam com eps=1e-5)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,
            eta_min=1e-5
        )
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: dict, iwae_samples: int = 1) -> dict:
        self.model.train()
        
        # Move to device
        obs = {k: v.to(self.device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(self.device)
        reset = batch['reset'].to(self.device)
        
        # Forward
        output = self.model(
            obs=obs,
            actions=actions,
            reset=reset,
            in_state=None,
            iwae_samples=iwae_samples,
            do_open_loop=False
        )
        
        # Backward
        self.optimizer.zero_grad()
        output['loss'].backward()
        
        # Gradient clipping (DreamerV2 usa 100)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Adiciona grad norm às métricas
        output['metrics']['grad_norm'] = grad_norm
        output['metrics']['lr'] = self.scheduler.get_last_lr()[0]
        
        return output
    
    def train_epoch(
        self,
        dataloader,
        iwae_samples: int = 1,
        log_interval: int = 10
    ):
        
        epoch_metrics = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Train step
            output = self.train_step(batch, iwae_samples=iwae_samples)
            
            # Accumulate metrics
            for key, value in output['metrics'].items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Log to tensorboard
            if self.global_step % log_interval == 0:
                for key, value in output['metrics'].items():
                    self.writer.add_scalar(
                        f'train/{key}',
                        value.item() if torch.is_tensor(value) else value,
                        self.global_step
                    )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': output['metrics']['loss_total'].item(),
                'kl': output['metrics']['loss_kl'].item(),
                'recon': output['metrics']['loss_reconstr'].item()
            })
            
            self.global_step += 1
        
        # Average epoch metrics
        epoch_metrics_avg = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        return epoch_metrics_avg
    
    @torch.no_grad()
    def validate(self, dataloader, iwae_samples: int = 1):
        self.model.eval()
        
        val_metrics = {}
        
        for batch in dataloader:
            # Move to device
            obs = {k: v.to(self.device) for k, v in batch['obs'].items()}
            actions = batch['actions'].to(self.device)
            reset = batch['reset'].to(self.device)
            
            # Forward
            output = self.model(
                obs=obs,
                actions=actions,
                reset=reset,
                in_state=None,
                iwae_samples=iwae_samples,
                do_open_loop=False
            )
            
            # Accumulate
            for key, value in output['metrics'].items():
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Average
        val_metrics_avg = {
            key: sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        
        # Log
        for key, value in val_metrics_avg.items():
            self.writer.add_scalar(f'val/{key}', value, self.global_step)
        
        return val_metrics_avg
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Loaded checkpoint from {filepath} (epoch {self.epoch})")
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = 100,
        iwae_samples: int = 1,
        save_interval: int = 10,
        log_interval: int = 10
    ):
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(
                train_dataloader,
                iwae_samples=iwae_samples,
                log_interval=log_interval
            )
            
            print(f"\nEpoch {epoch} - Train metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Validate
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader, iwae_samples=iwae_samples)
                print(f"\nEpoch {epoch} - Val metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Check if best
                val_loss = val_metrics['loss_total']
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print("\nTraining complete!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train AudioWorldModel')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with tokenized data')
    parser.add_argument('--val-data-dir', type=str, default=None,
                        help='Directory with validation data')
    
    # Model
    parser.add_argument('--vocab-size', type=int, default=512)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--deter-dim', type=int, default=512)
    parser.add_argument('--stoch-dim', type=int, default=32)
    parser.add_argument('--stoch-discrete', type=int, default=32)
    parser.add_argument('--encoder-out-dim', type=int, default=256)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=50)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--iwae-samples', type=int, default=1,
                        help='Number of IWAE samples (1 = standard ELBO)')
    parser.add_argument('--use-embeddings', action='store_true',
                        help='Use continuous embeddings instead of indices')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='runs/audio_world_model')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    
    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Config
    config = AudioWorldModelConfig(
        spectrogram_vocab_size=args.vocab_size,
        spectrogram_embed_dim=args.embed_dim,
        spectrogram_out_dim=args.encoder_out_dim,
        use_pretrained_embeddings=args.use_embeddings,
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        stoch_discrete=args.stoch_discrete
    )
    
    # Model
    model = AudioWorldModel(config)
    
    # Trainer
    trainer = AudioWorldModelTrainer(
        model=model,
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume if needed
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Data
    print("Loading data...")
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        use_embeddings=args.use_embeddings,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = None
    if args.val_data_dir:
        val_dataloader = create_dataloader(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            use_embeddings=args.use_embeddings,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        iwae_samples=args.iwae_samples,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )


if __name__ == '__main__':
    main()
