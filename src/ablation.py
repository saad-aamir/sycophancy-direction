"""Directional ablation hooks (Arditi et al. style).

For unit direction r_hat, every hooked activation x becomes
    x <- x - (x . r_hat) r_hat
i.e. the component along r_hat is projected out at all positions.

Hook coverage when cfg['ablation']['everywhere'] is true:
  - blocks.{l}.hook_resid_pre   for every layer
  - blocks.{l}.hook_resid_mid   for every layer   (catches attn writes)
  - blocks.{L-1}.hook_resid_post                  (catches final MLP write)
resid_post[l] == resid_pre[l+1], so intermediate resid_post hooks would be
redundant; the final layer's is the one with no following resid_pre.
"""
from __future__ import annotations

import torch

from .common import ROOT, load_config


def make_hook(direction: torch.Tensor):
    def hook(activation, hook=None):
        d = direction.to(activation.device, activation.dtype)
        proj = activation @ d                       # [batch, pos]
        return activation - proj.unsqueeze(-1) * d  # [batch, pos, d_model]
    return hook


def get_ablation_hooks(model, direction: torch.Tensor, everywhere: bool = True):
    fn = make_hook(direction / direction.norm())
    L = model.cfg.n_layers
    hooks = [(f"blocks.{l}.hook_resid_pre", fn) for l in range(L)]
    if everywhere:
        hooks += [(f"blocks.{l}.hook_resid_mid", fn) for l in range(L)]
        hooks += [(f"blocks.{L - 1}.hook_resid_post", fn)]
    return hooks


def load_ablation_hooks(model, cfg=None, pair_type: str | None = None):
    """Load the saved primary direction and return ready-to-use hooks."""
    cfg = cfg or load_config()
    pair_type = pair_type or cfg["extraction"]["pair_type"]
    blob = torch.load(
        ROOT / cfg["paths"]["directions_dir"] / f"{pair_type}_directions.pt"
    )
    layer = cfg["ablation"]["layer"]
    layer = blob["best_layer"] if layer == "auto" else int(layer)
    direction = blob["dirs"][layer]
    print(f"[ablation] using {pair_type} direction from layer {layer} "
          f"(AUROC {blob['auroc'][layer]:.3f})")
    return get_ablation_hooks(model, direction, cfg["ablation"]["everywhere"])
