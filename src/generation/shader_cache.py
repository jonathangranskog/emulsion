"""
Caching system for generated shaders to persist across sessions.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path


class ShaderCacheManager:
    """Manages persistent cache of generated shaders."""

    # Increment this when validation logic changes to invalidate old caches
    CACHE_VERSION = 2  # Version 2: Added texture name and pattern validation

    def __init__(self, cache_dir: str = ".cache/t2s_cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cached shaders (relative to cwd)
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created shader cache directory: {self.cache_dir}")

    def _prompt_to_hash(self, prompt: str) -> str:
        """
        Convert prompt to a safe filename using SHA256 hash.

        Args:
            prompt: The text prompt

        Returns:
            Hexadecimal hash string (first 16 chars for brevity)
        """
        prompt_normalized = prompt.strip().lower()
        hash_obj = hashlib.sha256(prompt_normalized.encode("utf-8"))
        return hash_obj.hexdigest()[:16]

    def _get_cache_path(self, prompt: str) -> Path:
        """
        Get the cache file path for a given prompt.

        Args:
            prompt: The text prompt

        Returns:
            Path to the cache file
        """
        prompt_hash = self._prompt_to_hash(prompt)
        return self.cache_dir / f"{prompt_hash}.json"

    def get_cached_shader(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached shader for the given prompt.

        Args:
            prompt: The text prompt

        Returns:
            Shader dictionary if found, None otherwise
        """
        cache_path = self._get_cache_path(prompt)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)

            # Check cache version - invalidate if outdated
            cached_version = cached_data.get("version", 1)
            if cached_version < self.CACHE_VERSION:
                print(
                    f"⚠️  Cache outdated (v{cached_version} < v{self.CACHE_VERSION}) "
                    f"for prompt '{prompt[:30]}...'. Regenerating."
                )
                # Delete outdated cache file
                cache_path.unlink()
                return None

            # Verify the prompt matches (hash collision check)
            if cached_data.get("prompt", "").strip().lower() != prompt.strip().lower():
                print(f"⚠️  Cache hash collision detected for prompt: {prompt[:30]}...")
                return None

            # Return the shader data
            shader_result = {
                "shader_code": cached_data["shader_code"],
                "parameters": cached_data["parameters"],
                "description": cached_data.get("description", ""),
                "success": True,
            }

            print(
                f"✨ Loaded cached shader (v{cached_version}) for: '{prompt[:30]}...'"
            )
            return shader_result

        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"⚠️  Error reading cache for prompt '{prompt[:30]}...': {e}")
            return None

    def cache_shader(self, prompt: str, shader_result: Dict[str, Any]):
        """
        Cache a successfully generated shader.

        Args:
            prompt: The text prompt
            shader_result: The generated shader dictionary
        """
        cache_path = self._get_cache_path(prompt)

        try:
            cache_data = {
                "prompt": prompt,
                "shader_code": shader_result["shader_code"],
                "parameters": shader_result["parameters"],
                "description": shader_result.get("description", ""),
                "version": self.CACHE_VERSION,  # Use current version
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(
                f"💾 Cached shader (v{self.CACHE_VERSION}) for prompt: '{prompt[:30]}...'"
            )

        except (IOError, TypeError) as e:
            print(f"⚠️  Error caching shader for prompt '{prompt[:30]}...': {e}")

    def clear_cache(self):
        """Delete all cached shaders."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            print(f"🗑️  Cleared all cached shaders")
        except Exception as e:
            print(f"⚠️  Error clearing cache: {e}")

    def list_cached_prompts(self) -> list[Dict[str, Any]]:
        """
        List all cached prompts with metadata.

        Returns:
            List of dicts with keys: prompt, hash, description
        """
        cached_prompts = []

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)

                cached_prompts.append(
                    {
                        "prompt": cached_data.get("prompt", "Unknown"),
                        "hash": cache_file.stem,
                        "description": cached_data.get("description", ""),
                    }
                )

        except Exception as e:
            print(f"⚠️  Error listing cached prompts: {e}")

        return cached_prompts

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "num_cached_shaders": len(cache_files),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir.absolute()),
        }
