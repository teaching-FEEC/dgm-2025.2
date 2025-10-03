# ─────────────────────────────────────────────────────────────
# File: src/metrics/synth_metrics.py
# Simple aggregator for SSIM / PSNR / SAM metrics.
# ─────────────────────────────────────────────────────────────

import torch
from torch import nn
from torchmetrics.image import  StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, SpectralAngleMapper



class _NoOpMetric(nn.Module):
    """ metric that always returns 0 and does nothing."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @torch.no_grad()
    def forward(self, *_args, **_kwargs) -> torch.Tensor:
        return torch.tensor(0.0)

    @torch.no_grad()
    def update(self, *_args, **_kwargs) -> None:
        return None

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def reset(self) -> None:
        return None


class SynthMetrics(nn.Module):
    """
    Aggregate SSIM, PSNR, and/or SAM 

    Example:
        metrics = SynthMetrics(metrics=['ssim', 'psnr'], data_range=1.0)
        out = metrics(pred, target)   # dict with values
        metrics.update(pred, target)
        agg = metrics.compute()
    """

    def __init__(self, metrics: list[str] = ('ssim', 'psnr', 'sam'), data_range: float = 1.0):
        super().__init__()
        self.enabled = set(m.lower() for m in metrics)
        self.data_range = data_range

        if 'ssim' in self.enabled:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range)
        else:
            self.ssim = _NoOpMetric('ssim')

        if 'psnr' in self.enabled:
            self.psnr = PeakSignalNoiseRatio(data_range=self.data_range)
        else:
            self.psnr = _NoOpMetric('psnr')

        if 'sam' in self.enabled:
            self.sam = SpectralAngleMapper()
        else:
            self.sam = _NoOpMetric('sam')

        self._order = [name for name in ('ssim', 'psnr', 'sam') if name in self.enabled]

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        if 'ssim' in self.enabled:
            results['ssim'] = self.ssim(pred, target)
        if 'psnr' in self.enabled:
            results['psnr'] = self.psnr(pred, target)
        if 'sam' in self.enabled:
            results['sam'] = self.sam(pred, target)
        return {k: results[k] for k in self._order}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if 'ssim' in self.enabled:
            self.ssim.update(pred, target)
        if 'psnr' in self.enabled:
            self.psnr.update(pred, target)
        if 'sam' in self.enabled:
            self.sam.update(pred, target)

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if 'ssim' in self.enabled:
            out['ssim'] = self.ssim.compute()
        if 'psnr' in self.enabled:
            out['psnr'] = self.psnr.compute()
        if 'sam' in self.enabled:
            out['sam'] = self.sam.compute()
        return {k: out[k] for k in self._order}

    def reset(self) -> None:
        if 'ssim' in self.enabled:
            self.ssim.reset()
        if 'psnr' in self.enabled:
            self.psnr.reset()
        if 'sam' in self.enabled:
            self.sam.reset()
