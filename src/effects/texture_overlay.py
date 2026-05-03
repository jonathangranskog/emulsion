"""Texture overlay effect with multiple blend modes."""

from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

from src.effects.base import ImageEffect


class TextureOverlay(ImageEffect):
    """Apply a texture overlay with various blend modes.

    Supports blend modes: Normal, Multiply, Screen, Overlay, Add, Subtract
    If the texture has an alpha channel, it will be used for transparency.
    """

    BLEND_MODES = ["Normal", "Multiply", "Screen", "Overlay", "Add", "Subtract"]

    def __init__(
        self,
        texture_path: str = "",
        blend_mode: int = 0,
        opacity: float = 1.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        flip_x: bool = False,
        flip_y: bool = False,
        use_bw: bool = False,
    ):
        super().__init__()
        self.texture_path = texture_path
        self.blend_mode = max(0, min(blend_mode, len(self.BLEND_MODES) - 1))
        self.opacity = opacity
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.use_bw = use_bw
        self._texture_tensor = None
        self._texture_loaded = False

    def _load_texture(self) -> torch.Tensor:
        """Load texture from file path."""
        if not self.texture_path or not Path(self.texture_path).exists():
            # Return a 1x1 transparent texture if no path is set (CHW format)
            return torch.zeros(4, 1, 1)

        try:
            img = Image.open(self.texture_path)

            # Convert to RGBA to ensure we have alpha channel
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Convert to numpy array [H, W, 4] and normalize to [0, 1]
            texture = torch.from_numpy(np.array(img)).float() / 255.0

            # Permute to PyTorch format [C, H, W]
            texture = texture.permute(2, 0, 1)

            return texture

        except Exception as e:
            print(f"Failed to load texture: {e}")
            # Return transparent fallback on error
            return torch.zeros(4, 1, 1)

    def on_file_load(self, texture_path: str):
        """Called when a texture file is loaded through the UI."""
        print(f"Loading texture from {texture_path}")
        if not Path(texture_path).exists():
            print(f"Texture file {texture_path} does not exist")
            return

        self.texture_path = texture_path
        self._texture_loaded = False  # Force reload
        self._texture_tensor = None

    def serialize_to_cache(self) -> Dict[str, Any]:
        """
        Serialize TextureOverlay state including the loaded texture data.

        Overrides base implementation to include the loaded texture tensor
        to preserve the texture when restoring from cache.
        """
        state = super().serialize_to_cache()
        # Store the texture data if it's been loaded
        state["texture_path"] = self.texture_path
        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "TextureOverlay":
        """
        Deserialize TextureOverlay from cached state.

        Overrides base implementation to properly restore the loaded texture data.
        """
        # Create effect with all parameters
        effect = cls(
            texture_path=state.get("texture_path", ""),
            blend_mode=state.get("blend_mode", 0),
            opacity=state.get("opacity", 1.0),
            scale_x=state.get("scale_x", 1.0),
            scale_y=state.get("scale_y", 1.0),
            flip_x=state.get("flip_x", False),
            flip_y=state.get("flip_y", False),
            use_bw=state.get("use_bw", False),
        )

        if effect.texture_path != "":
            effect.on_file_load(effect.texture_path)
        return effect

    def sync_unexposed_parameters(self, other: "TextureOverlay") -> bool:
        """Sync loaded texture tensor from another TextureOverlay effect."""
        changed = False
        if other._texture_tensor is not None:
            if self._texture_tensor is None or not torch.equal(
                self._texture_tensor, other._texture_tensor
            ):
                self._texture_tensor = other._texture_tensor.clone()
                self._texture_loaded = other._texture_loaded
                changed = True
        return changed

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "texture_path",
                "label": "Texture File",
                "type": "file",
                "default": "",
                "file_types": [
                    ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")
                ],
            },
            {
                "name": "blend_mode",
                "label": "Blend Mode",
                "type": "choice",
                "default": 0,
                "choices": self.BLEND_MODES,
            },
            {
                "name": "opacity",
                "label": "Opacity",
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "scale_x",
                "label": "Scale X",
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 1.5,
                "step": 0.1,
            },
            {
                "name": "scale_y",
                "label": "Scale Y",
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 1.5,
                "step": 0.1,
            },
            {
                "name": "flip_x",
                "label": "Flip X",
                "type": "bool",
                "default": False,
            },
            {
                "name": "flip_y",
                "label": "Flip Y",
                "type": "bool",
                "default": False,
            },
            {
                "name": "use_bw",
                "label": "Black & White",
                "type": "bool",
                "default": False,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply texture overlay using torch operations."""
        # Load texture if needed
        if self._texture_tensor is None or not self._texture_loaded:
            self._texture_tensor = self._load_texture()
            self._texture_loaded = True

        if self._texture_tensor is None:
            return x

        # x is [C, H, W], texture is [4, H_tex, W_tex]
        C, H, W = x.shape
        texture = self._texture_tensor

        # Ensure texture is on the same device as input
        texture = texture.to(x.device)

        tex_h, tex_w = texture.shape[1], texture.shape[2]

        # Calculate automatic scale to cover the image
        # Use max to ensure texture covers entire image (no gaps)
        scale_h = H / tex_h
        scale_w = W / tex_w
        auto_scale = max(scale_h, scale_w)

        # Apply user scale on top of automatic scale (separately for x and y)
        final_scale_h = auto_scale * self.scale_y
        final_scale_w = auto_scale * self.scale_x

        # Calculate target dimensions
        target_h = int(tex_h * final_scale_h)
        target_w = int(tex_w * final_scale_w)

        # Scale the texture
        texture = texture.unsqueeze(0)
        texture = F.interpolate(
            texture, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        texture = texture.squeeze(0)

        # Apply flipping
        if self.flip_y:
            texture = torch.flip(texture, [1])  # Flip height dimension
        if self.flip_x:
            texture = torch.flip(texture, [2])  # Flip width dimension

        # Update dimensions after scaling
        tex_h, tex_w = target_h, target_w

        # Tile texture to cover the image if needed
        if tex_h < H or tex_w < W:
            tiles_h = (H + tex_h - 1) // tex_h
            tiles_w = (W + tex_w - 1) // tex_w
            texture = texture.repeat(1, tiles_h, tiles_w)
            tex_h, tex_w = texture.shape[1], texture.shape[2]

        # Crop to match image size (centered)
        if tex_h > H or tex_w > W:
            offset_h = (tex_h - H) // 2
            offset_w = (tex_w - W) // 2
            texture = texture[:, offset_h : offset_h + H, offset_w : offset_w + W]
        else:
            # Should match exactly at this point, but handle edge cases
            texture = texture[:, :H, :W]

        # Extract RGB and alpha channels
        # texture is [4, H, W], split into RGB [3, H, W] and alpha [1, H, W]
        texture_rgb = texture[:3, :, :]
        texture_alpha = texture[3:4, :, :]

        # Convert to grayscale if requested
        if self.use_bw:
            # Standard luminance weights: Rec. 709
            luminance = (
                0.2126 * texture_rgb[0:1, :, :]
                + 0.7152 * texture_rgb[1:2, :, :]
                + 0.0722 * texture_rgb[2:3, :, :]
            )
            # Replicate luminance across RGB channels
            texture_rgb = luminance.repeat(3, 1, 1)

        # Ensure x has the right number of channels
        if C == 1:
            # Grayscale input - expand to RGB
            x_rgb = x.repeat(3, 1, 1)
        elif C >= 3:
            x_rgb = x[:3, :, :]
        else:
            return x

        # Apply blend mode based on index
        if self.blend_mode == 0:  # Normal
            blended = texture_rgb
        elif self.blend_mode == 1:  # Multiply
            blended = x_rgb * texture_rgb
        elif self.blend_mode == 2:  # Screen
            blended = 1.0 - (1.0 - x_rgb) * (1.0 - texture_rgb)
        elif self.blend_mode == 3:  # Overlay
            # Overlay: multiply if base < 0.5, screen if base >= 0.5
            blended = torch.where(
                x_rgb < 0.5,
                2.0 * x_rgb * texture_rgb,
                1.0 - 2.0 * (1.0 - x_rgb) * (1.0 - texture_rgb),
            )
        elif self.blend_mode == 4:  # Add
            blended = x_rgb + texture_rgb
        elif self.blend_mode == 5:  # Subtract
            blended = x_rgb - texture_rgb
        else:
            blended = texture_rgb

        # Mix based on texture alpha and opacity
        alpha = texture_alpha * self.opacity
        result = x_rgb * (1.0 - alpha) + blended * alpha

        # Handle alpha channel in output if present
        if C == 4:
            result = torch.cat([result, x[3:4, :, :]], dim=0)
        elif C == 1:
            # Convert back to grayscale
            result = 0.299 * result[0:1] + 0.587 * result[1:2] + 0.114 * result[2:3]

        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Return GLSL shader code and uniforms."""
        # Load texture if needed
        if self._texture_tensor is None or not self._texture_loaded:
            self._texture_tensor = self._load_texture()
            self._texture_loaded = True

        glsl_code = """
        vec4 apply_textureoverlay(vec4 color, int blend_mode, int flip_x, int flip_y, float opacity, sampler2D overlay_texture, float scale_x, float scale_y, int use_bw) {
            vec2 main_size = textureSize(main_texture, 0);
            vec2 overlay_size = textureSize(overlay_texture, 0);

            // Calculate automatic scale to cover the image
            // Use max to ensure texture covers entire image (no gaps)
            float scale_h = main_size.y / overlay_size.y;
            float scale_w = main_size.x / overlay_size.x;
            float auto_scale = max(scale_h, scale_w);

            // Apply user scale on top of automatic scale (separately for x and y)
            float final_scale_h = auto_scale * scale_y;
            float final_scale_w = auto_scale * scale_x;

            // Calculate scaled overlay size
            vec2 scaled_overlay_size = vec2(overlay_size.x * final_scale_w, overlay_size.y * final_scale_h);

            // Calculate UV coordinates
            vec2 pixel_coord = TexCoord * main_size;
            vec2 overlay_uv = pixel_coord / scaled_overlay_size;

            // Handle cropping (center the texture)
            vec2 offset = (scaled_overlay_size - main_size) / (2.0 * scaled_overlay_size);
            overlay_uv = overlay_uv + offset;

            // Apply flipping
            if (flip_x != 0) {
                overlay_uv.x = 1.0 - overlay_uv.x;
            }
            if (flip_y != 0) {
                overlay_uv.y = 1.0 - overlay_uv.y;
            }

            // Sample the overlay texture
            vec4 overlay = texture(overlay_texture, overlay_uv);

            // Extract RGB and alpha
            vec3 base = color.rgb;
            vec3 tex_rgb = overlay.rgb;

            // Convert to grayscale if requested
            if (use_bw != 0) {
                // Standard luminance weights: Rec. 709
                float luminance = 0.2126 * tex_rgb.r + 0.7152 * tex_rgb.g + 0.0722 * tex_rgb.b;
                tex_rgb = vec3(luminance);
            }
            float tex_alpha = overlay.a * opacity;

            // Apply blend mode
            vec3 blended;
            if (blend_mode == 0) {  // Normal
                blended = tex_rgb;
            } else if (blend_mode == 1) {  // Multiply
                blended = base * tex_rgb;
            } else if (blend_mode == 2) {  // Screen
                blended = vec3(1.0) - (vec3(1.0) - base) * (vec3(1.0) - tex_rgb);
            } else if (blend_mode == 3) {  // Overlay
                blended = mix(
                    2.0 * base * tex_rgb,
                    vec3(1.0) - 2.0 * (vec3(1.0) - base) * (vec3(1.0) - tex_rgb),
                    step(0.5, base)
                );
            } else if (blend_mode == 4) {  // Add
                blended = base + tex_rgb;
            } else if (blend_mode == 5) {  // Subtract
                blended = base - tex_rgb;
            } else {
                blended = tex_rgb;
            }

            // Mix based on texture alpha
            vec3 result = mix(base, blended, tex_alpha);

            return vec4(result, color.a);
        }
        """

        uniforms = {
            "u_overlay_texture": self._texture_tensor,
            "u_blend_mode": self.blend_mode,
            "u_flip_x": self.flip_x,
            "u_flip_y": self.flip_y,
            "u_opacity": self.opacity,
            "u_scale_x": self.scale_x,
            "u_scale_y": self.scale_y,
            "u_use_bw": self.use_bw,
        }

        return [glsl_code], uniforms
