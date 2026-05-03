"""Disk-based texture cache for effects."""

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TextureDiskCache:
    """Manages caching of effect textures to disk."""

    CACHE_DIR = Path(".cache/textures")

    @classmethod
    def _ensure_cache_dir(cls):
        """Ensure the cache directory exists."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _compute_hash(cls, effect_name: str, params: dict) -> str:
        """
        Compute a hash from effect name and parameters for deduplication.

        Args:
            effect_name: Name of the effect (e.g., "GaussianGrain")
            params: Dictionary of parameters used to generate the texture

        Returns:
            Hash string to use as filename
        """
        # Create a stable string representation of params
        param_str = f"{effect_name}:" + ":".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        # Compute hash
        hash_obj = hashlib.md5(param_str.encode())
        return hash_obj.hexdigest()[:16]

    @classmethod
    def get_cache_path(cls, effect_name: str, params: dict) -> Path:
        """Get the expected cache file path for the given effect name and params."""
        param_hash = cls._compute_hash(effect_name, params)
        filename = f"{effect_name.lower()}_{param_hash}.npy"
        return cls.CACHE_DIR / filename

    @classmethod
    def save_texture(cls, texture: torch.Tensor, effect_name: str, params: dict) -> str:
        """
        Save a texture to disk as uncompressed NPY.

        Args:
            texture: Tensor of shape (C, H, W) with values in [0, 1] or [-1, 1]
            effect_name: Name of the effect
            params: Parameters used to generate the texture (for hashing)

        Returns:
            Relative path to the saved texture file
        """
        cls._ensure_cache_dir()

        # Compute filename from hash
        param_hash = cls._compute_hash(effect_name, params)
        filename = f"{effect_name.lower()}_{param_hash}.npy"
        filepath = cls.CACHE_DIR / filename

        # Check if already exists (deduplication)
        if filepath.exists():
            return str(filepath)

        # Convert tensor to numpy (C, H, W) preserving float32 and negative values
        texture_np = texture.cpu().numpy()

        # Save as uncompressed NPY (much faster than compressed NPZ)
        # Grain textures are random noise and don't compress well anyway
        np.save(filepath, texture_np)

        return str(filepath)

    @classmethod
    def load_texture(cls, filepath: str) -> Optional[torch.Tensor]:
        """
        Load a texture from disk.

        Args:
            filepath: Path to the texture file (.npy)

        Returns:
            Tensor of shape (C, H, W) preserving original values (including negatives), or None if file doesn't exist
        """
        path = Path(filepath)
        if not path.exists():
            return None

        try:
            # Load uncompressed NPY file (fast)
            texture_np = np.load(path)

            # Convert to torch tensor, preserving float32 and negative values
            texture = torch.from_numpy(texture_np).float()

            return texture
        except Exception as e:
            print(f"Warning: Failed to load texture from {filepath}: {e}")
            return None
