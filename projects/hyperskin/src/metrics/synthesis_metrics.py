# ─────────────────────────────────────────────────────────────
# File: src/metrics/synth_metrics.py
#
# Unified wrapper for synthesis metrics: SSIM, PSNR, SAM, and FID.
# This version **explicitly** uses your Inception wrapper located at:
#     from src.metrics.inception import InceptionV3Wrapper
#
# Key details:
#   • Never changes your VAE / FastGAN modules or YAML configs.
#   • FID is lazily constructed on first real use (Lightning/CLI friendly).
#   • FID inputs are always float32 on [0,1] and moved to the FID device.
#   • Heuristic normalizer maps [-1,1] → [0,1] when detected; otherwise
#     divides by `data_range` if it’s not 1.0.
#   • Minimal, efficient code; each line commented for clarity.
# ─────────────────────────────────────────────────────────────

# ------------------------------
# Standard library imports
# ------------------------------
from __future__ import annotations               # Enable modern type-hints on older Python
import warnings                                  # Soft warnings instead of hard errors

# ------------------------------
# PyTorch imports
# ------------------------------
import torch                                     # Core tensor library
from torch import nn                             # Base class for neural modules

# ------------------------------
# TorchMetrics imports
# ------------------------------
from torchmetrics.image import (                 # Image quality metrics collection
    StructuralSimilarityIndexMeasure,            # SSIM metric
    PeakSignalNoiseRatio,                        # PSNR metric
    SpectralAngleMapper,                         # SAM metric
)
from torchmetrics.image.fid import FrechetInceptionDistance  # FID metric

# ------------------------------
# Your explicit Inception wrapper path (requested)
# ------------------------------
from src.metrics.inception import InceptionV3Wrapper         # <- fixed import path


# ─────────────────────────────────────────────────────────────
# No-op metric used when a metric is disabled or cannot be built yet
# ─────────────────────────────────────────────────────────────
class _NoOpMetric(nn.Module):
    """Metric stub that behaves like TorchMetrics but always returns 0."""

    def __init__(self, name: str):
        super().__init__()                                   # Initialize nn.Module
        self.name = name                                     # Store metric name for debugging

    @torch.no_grad()
    def forward(self, *_args, **_kwargs) -> torch.Tensor:
        return torch.tensor(0.0)                             # Always return scalar zero

    @torch.no_grad()
    def update(self, *_args, **_kwargs) -> None:
        return None                                          # No state to update

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return torch.tensor(0.0)                             # Always compute zero

    def reset(self) -> None:
        return None                                          # Nothing to reset


# ─────────────────────────────────────────────────────────────
# Main aggregator for SSIM / PSNR / SAM / FID
# ─────────────────────────────────────────────────────────────
class SynthMetrics(nn.Module):
    """
    Unified interface for synthesis metrics.

    Args:
        metrics:
            List of metric names to enable. Any subset of ['ssim', 'psnr', 'sam', 'fid'].
        data_range:
            Expected range of input tensors *before* any normalization here.
            - If your tensors are in [-1, 1], keep `data_range=1.0` and we auto-map to [0,1].
            - If your tensors are already [0,1], keep `data_range=1.0` (no change).
            - If your tensors are [0,255], set `data_range=255.0` (we map to [0,1]).
        fid_in_chans:
            Optional override for channel count used by the Inception wrapper.
            If None, inferred from incoming tensors on first use.
        fid_im_size:
            Optional override for spatial size used by the Inception wrapper.
            Accepts an int (square) or (H, W). If None, inferred from incoming tensors.
        fid_normalize_input:
            Passed to InceptionV3Wrapper(normalize_input=...). Keep False to match FastGAN style.
    """

    def __init__(
        self,
        metrics: list[str] = ('ssim', 'psnr', 'sam'),        # Enabled metrics by default
        data_range: float = 1.0,                             # Expected tensor range pre-normalization
        *,
        fid_in_chans: int | None = None,                     # Optional explicit channels for FID
        fid_im_size: int | tuple[int, int] | None = None,    # Optional explicit spatial size for FID
        fid_normalize_input: bool = False,                   # Whether the wrapper normalizes internally
    ):
        super().__init__()                                   # Initialize nn.Module

        # Normalize metric names (case-insensitive) and store in a set for fast checks
        self.enabled = set(m.lower() for m in metrics)       # e.g., {'ssim','psnr','sam','fid'}

        # Persist the provided data range for later normalization logic
        self.data_range = float(data_range)                  # e.g., 1.0 or 255.0

        # Instantiate real metrics or no-op stubs depending on `enabled`
        self.ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range) if 'ssim' in self.enabled else _NoOpMetric('ssim')  # SSIM
        self.psnr = PeakSignalNoiseRatio(data_range=self.data_range)             if 'psnr' in self.enabled else _NoOpMetric('psnr')  # PSNR
        self.sam  = SpectralAngleMapper()                                        if 'sam'  in self.enabled else _NoOpMetric('sam')   # SAM

        # FID starts as No-Op; we lazily build it on first real use to avoid CLI-time errors
        self.fid: nn.Module = _NoOpMetric('fid')                                 # Placeholder until built

        # Save FID construction knobs for lazy build
        self._fid_cfg = {
            "enabled": ('fid' in self.enabled),                                   # Whether FID is requested
            "fid_in_chans": fid_in_chans,                                         # Possibly None (infer later)
            "fid_im_size": fid_im_size,                                           # Possibly None (infer later)
            "fid_normalize_input": bool(fid_normalize_input),                     # Wrapper flag
            "built": False,                                                       # Flip to True only on success
        }

        # Define the stable order of keys in returned dictionaries
        self._order = [name for name in ('ssim', 'psnr', 'sam', 'fid') if name in self.enabled]  # Respect requested order

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _map_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert arbitrary inputs to float32 in [0,1] for FID.

        Heuristic:
          • If the tensor contains negatives, we assume a [-1,1] domain and map as (x+1)/2.
          • Else if data_range != 1.0, we divide by data_range (e.g., [0,255] → [0,1]).
          • Else we assume already [0,1].
        """
        x = x.detach().to(dtype=torch.float32)                                      # Ensure float32 and no-grad
        if torch.min(x).item() < -1e-6:                                             # Detect presence of negatives
            x = (x + 1.0) * 0.5                                                     # Map [-1,1] → [0,1]
        elif self.data_range != 1.0:                                                # Otherwise, use given scale
            x = x / self.data_range                                                 # Map [0,data_range] → [0,1]
        return x.clamp_(0.0, 1.0)                                                   # Clamp for safety

    def _fid_device(self) -> torch.device:
        """
        Determine the device hosting the FID metric (params or buffers).
        Falls back to CPU if none exist yet.
        """
        for p in self.fid.parameters() if hasattr(self.fid, "parameters") else []:  # Inspect parameters first
            return p.device                                                         # Return the first parameter's device
        for b in self.fid.buffers() if hasattr(self.fid, "buffers") else []:        # Then buffers (states)
            return b.device                                                         # Return the first buffer's device
        return torch.device("cpu")                                                  # Default to CPU

    def _maybe_build_fid(self, tensor_for_shape: torch.Tensor) -> None:
        """
        Lazily instantiate the FID metric with InceptionV3Wrapper once a real batch is seen.
        Never raises in order to remain CLI-friendly; if shape is not inferable, it tries again later.
        """
        if not self._fid_cfg["enabled"] or self._fid_cfg["built"]:                  # Skip if not requested or already built
            return                                                                  # Nothing to do

        try:
            _, c, h, w = tensor_for_shape.shape                                     # Expect 4D (N,C,H,W)
        except Exception:
            return                                                                  # Can't infer yet; try again on next call

        in_chans = int(self._fid_cfg["fid_in_chans"]) if self._fid_cfg["fid_in_chans"] is not None else int(c)  # Channels (explicit or inferred)

        # Resolve spatial size: explicit overrides inference
        if self._fid_cfg["fid_im_size"] is None:                                    # If not provided
            input_img_size = (in_chans, int(h), int(w))                             # Use current tensor size
        else:                                                                       
            if isinstance(self._fid_cfg["fid_im_size"], int):                       # Square spatial size
                s = int(self._fid_cfg["fid_im_size"])                               # Cast to int
                input_img_size = (in_chans, s, s)                                   # (C,S,S)
            else:                                                                   
                hh, ww = self._fid_cfg["fid_im_size"]                               # Tuple (H,W)
                input_img_size = (in_chans, int(hh), int(ww))                       # (C,H,W)

        # Instantiate your Inception wrapper with the agreed parameters
        inception = InceptionV3Wrapper(                                             # ← explicit import path
            normalize_input=self._fid_cfg["fid_normalize_input"],                   # Pass-through knob
            in_chans=in_chans,                                                      # Number of channels
        )
        inception.eval()                                                            # Use eval mode for deterministic metrics

        # Create the FID metric bound to the wrapper
        fid_metric = FrechetInceptionDistance(
            inception,                                              # Explicit kwarg (TM new-style)
            input_img_size=input_img_size,                                          # (C,H,W) expected by your wrapper
        )
        fid_metric.eval()                                                           # Ensure eval mode

        # Swap the No-Op for the real metric and mark as built
        self.fid = fid_metric                                                       # Replace stub with actual metric
        self._fid_cfg["built"] = True                                               # Mark successful construction

    # ─────────────────────────────────────────────────────────
    # Public API: forward (one-shot evaluation)
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute metrics for a single (pred, target) batch.
        FID is computed in isolation (reset→update→compute→reset) so it does not
        interfere with any accumulated state from `update()`.
        """
        results: dict[str, torch.Tensor] = {}                                       # Output container

        if 'ssim' in self.enabled:                                                  # If SSIM requested
            results['ssim'] = self.ssim(pred, target)                               # Compute SSIM directly

        if 'psnr' in self.enabled:                                                  # If PSNR requested
            results['psnr'] = self.psnr(pred, target)                               # Compute PSNR directly

        if 'sam' in self.enabled:                                                   # If SAM requested
            results['sam'] = self.sam(pred, target)                                 # Compute SAM directly

        if 'fid' in self.enabled:                                                   # If FID requested
            self._maybe_build_fid(pred)                                             # Lazily build if needed
            if not isinstance(self.fid, _NoOpMetric):                               # Ensure we have a real FID
                dev = self._fid_device()                                            # Get FID's device
                real01 = self._map_to_unit_range(target).to(dev, non_blocking=True) # Map target to [0,1] and move device
                fake01 = self._map_to_unit_range(pred).to(dev, non_blocking=True)   # Map pred to [0,1] and move device
                self.fid.reset()                                                    # Isolate computation
                self.fid.update(real01, real=True)                                  # Feed real images
                self.fid.update(fake01, real=False)                                 # Feed fake images
                results['fid'] = self.fid.compute()                                 # Compute FID
                self.fid.reset()                                                    # Reset so forward() is side-effect free
            else:
                results['fid'] = torch.tensor(0.0)                                  # Not built yet → return 0

        return {k: results[k] for k in self._order}                                 # Preserve stable key order

    # ─────────────────────────────────────────────────────────
    # Public API: update (accumulate across batches)
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate metric states across multiple batches.
        This is the recommended path for epoch-level logging.
        """
        if 'ssim' in self.enabled:                                                  # SSIM accumulation
            self.ssim.update(pred, target)                                          # Update state

        if 'psnr' in self.enabled:                                                  # PSNR accumulation
            self.psnr.update(pred, target)                                          # Update state

        if 'sam' in self.enabled:                                                   # SAM accumulation
            self.sam.update(pred, target)                                           # Update state

        if 'fid' in self.enabled:                                                   # FID accumulation
            self._maybe_build_fid(pred)                                             # Build if not yet built
            if not isinstance(self.fid, _NoOpMetric):                               # If FID is real
                dev = self._fid_device()                                            # Find metric device
                real01 = self._map_to_unit_range(target).to(dev, non_blocking=True) # Normalize/move real
                fake01 = self._map_to_unit_range(pred).to(dev, non_blocking=True)   # Normalize/move fake
                self.fid.update(real01, real=True)                                  # Accumulate real
                self.fid.update(fake01, real=False)                                 # Accumulate fake
            # If still No-Op, silently skip (we avoid noisy logs during bootstrap)

    # ─────────────────────────────────────────────────────────
    # Public API: compute (finalize accumulated values)
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """
        Finalize and return accumulated metrics.
        If FID was never successfully built, it returns 0.0 for FID.
        """
        out: dict[str, torch.Tensor] = {}                                           # Aggregated results

        if 'ssim' in self.enabled:                                                  # SSIM finalize
            out['ssim'] = self.ssim.compute()                                       # Compute SSIM

        if 'psnr' in self.enabled:                                                  # PSNR finalize
            out['psnr'] = self.psnr.compute()                                       # Compute PSNR

        if 'sam' in self.enabled:                                                   # SAM finalize
            out['sam'] = self.sam.compute()                                         # Compute SAM

        if 'fid' in self.enabled:                                                   # FID finalize
            out['fid'] = self.fid.compute() if not isinstance(self.fid, _NoOpMetric) else torch.tensor(0.0)  # Compute or 0

        return {k: out[k] for k in self._order}                                     # Preserve stable key order

    # ─────────────────────────────────────────────────────────
    # Public API: reset states (useful between epochs)
    # ─────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset internal states for all enabled metrics."""
        if 'ssim' in self.enabled: self.ssim.reset()                                 # Clear SSIM state
        if 'psnr' in self.enabled: self.psnr.reset()                                 # Clear PSNR state
        if 'sam'  in self.enabled: self.sam.reset()                                  # Clear SAM state
        if 'fid'  in self.enabled and not isinstance(self.fid, _NoOpMetric):         # Clear FID state if active
            self.fid.reset()
