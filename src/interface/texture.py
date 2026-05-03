from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import OpenGL.GL as gl
import torch

from src.interface.error import check_for_errors


@dataclass
class Sampler2D:
    uniform_name: str
    id: int
    texture_unit: int
    resolution: tuple[int, int]
    channels: int


@dataclass
class Sampler3D:
    uniform_name: str
    id: int
    texture_unit: int
    resolution: tuple[int, int, int]
    channels: int


@dataclass
class FBO:
    fbo_name: str
    id: int
    resolution: tuple[int, int]
    texture_name: str


class TextureManager:
    _textures: dict[str, Any] = {}
    _fbo: dict[str, FBO] = {}
    _next_texture_unit: int = 1
    _freed_texture_units: list[int] = []  # Pool of freed texture units for reuse

    @classmethod
    def get_next_texture_unit(cls):
        # Reuse freed texture units first
        if cls._freed_texture_units:
            return cls._freed_texture_units.pop(0)

        # Otherwise allocate a new unit
        unit = cls._next_texture_unit
        cls._next_texture_unit += 1
        return unit

    @classmethod
    def free_texture_unit(cls, unit: int):
        """Return a texture unit to the pool for reuse"""
        if unit not in cls._freed_texture_units and unit > 0:
            cls._freed_texture_units.append(unit)

    @classmethod
    def upload_texture_tensor(cls, tensor: torch.Tensor, texture_name: str):
        ndim = tensor.ndim
        sampler = cls._textures.get(texture_name)
        if sampler is None:
            raise ValueError(f"Texture {texture_name} not found")
        # 2d texture
        if ndim == 3:
            gl.glActiveTexture(gl.GL_TEXTURE0 + sampler.texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, sampler.id)
            # Upload the tensor to the texture
            numpy_array = tensor.permute(1, 2, 0).numpy()

            # Determine format based on number of channels
            channels = tensor.shape[0]
            match channels:
                case 3:
                    internal_format = gl.GL_RGB32F
                    gl_format = gl.GL_RGB
                case 4:
                    internal_format = gl.GL_RGBA32F
                    gl_format = gl.GL_RGBA
                case _:
                    raise ValueError(
                        f"Unsupported number of channels: {channels}. Only RGB (3) and RGBA (4) are supported."
                    )

            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                internal_format,  # Use appropriate format for channel count
                numpy_array.shape[1],
                numpy_array.shape[0],
                0,
                gl_format,
                gl.GL_FLOAT,
                numpy_array,
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            # Update sampler resolution and channels if changed
            new_resolution = tensor.shape[1:3]  # (H, W)
            if sampler.resolution != new_resolution:
                sampler.resolution = new_resolution
            if sampler.channels != channels:
                sampler.channels = channels
        # 3d texture
        elif ndim == 4:
            gl.glActiveTexture(gl.GL_TEXTURE0 + sampler.texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_3D, sampler.id)
            # Upload the tensor to the texture
            numpy_array = tensor.numpy()
            gl.glTexImage3D(
                gl.GL_TEXTURE_3D,
                0,
                gl.GL_RGB32F,  # Use 32-bit float internal format for HDR/negative values
                # TODO: Is this right for 3d?
                numpy_array.shape[2],
                numpy_array.shape[1],
                numpy_array.shape[0],
                0,
                gl.GL_RGB,
                gl.GL_FLOAT,
                numpy_array,
            )
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

            # Update sampler resolution if dimensions changed
            new_resolution = tensor.shape[1:4]  # (D, H, W)
            if sampler.resolution != new_resolution:
                sampler.resolution = new_resolution
        else:
            raise ValueError(
                f"Unsupported tensor dimensions for texture '{texture_name}' upload: {ndim}"
            )
        gl.glActiveTexture(gl.GL_TEXTURE0)
        check_for_errors()

    @classmethod
    def register_texture2d(cls, tensor: torch.Tensor, texture_name: str):
        # Create the texture for opengl
        if texture_name not in cls._textures:
            texture_id = gl.glGenTextures(1)
            texture_unit = cls.get_next_texture_unit()
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            check_for_errors()

            # Create the sampler 2d for storage
            sampler2d = Sampler2D(
                uniform_name=texture_name,
                id=texture_id,
                resolution=tensor.shape[1:3],
                channels=tensor.shape[0],
                texture_unit=texture_unit,
            )
            cls._textures[sampler2d.uniform_name] = sampler2d
        else:
            print(f"Warning: Texture {texture_name} already registered")
        cls.upload_texture_tensor(tensor, texture_name)

    @classmethod
    def register_texture3d(cls, tensor: torch.Tensor, texture_name: str):
        # Create the texture for opengl
        if texture_name not in cls._textures:
            texture_id = gl.glGenTextures(1)
            texture_unit = cls.get_next_texture_unit()
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_3D, texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(
                gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE
            )
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            check_for_errors()
            sampler3d = Sampler3D(
                uniform_name=texture_name,
                id=texture_id,
                resolution=tensor.shape[1:4],
                channels=tensor.shape[0],
                texture_unit=texture_unit,
            )
            cls._textures[sampler3d.uniform_name] = sampler3d
        else:
            print(f"Warning: Texture {texture_name} already registered")
        cls.upload_texture_tensor(tensor, texture_name)

    @classmethod
    def register_fbo(cls, fbo_name: str, texture_name: str, width: int, height: int):
        # Create the FBO
        fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)

        # Create the texture (allocate texture unit once!)
        texture_id = gl.glGenTextures(1)
        texture_unit = cls.get_next_texture_unit()
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB32F,
            width,
            height,
            0,
            gl.GL_RGB,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture_id, 0
        )

        # Check FBO status
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer is not complete: {status}")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        check_for_errors()

        # Add the FBO to the manager
        fbo = FBO(
            fbo_name=fbo_name,
            id=fbo_id,
            resolution=(height, width),
            texture_name=texture_name,
        )
        cls._fbo[fbo.fbo_name] = fbo

        # Add the texture to the manager (use the same texture_unit we allocated)
        texture = Sampler2D(
            uniform_name=texture_name,
            id=texture_id,
            resolution=(height, width),
            channels=3,
            texture_unit=texture_unit,  # Use the unit we already allocated
        )
        cls._textures[texture.uniform_name] = texture

    @classmethod
    def delete_fbo(cls, fbo_name: str):
        fbo = cls._fbo.get(fbo_name)
        if fbo is None:
            raise ValueError(f"FBO {fbo_name} not found")
        gl.glDeleteFramebuffers(1, [fbo.id])
        del cls._fbo[fbo.fbo_name]
        cls.delete_texture(fbo.texture_name)
        check_for_errors()

    @classmethod
    def delete_texture(cls, texture_name: str):
        sampler = cls._textures.get(texture_name)
        if sampler is None:
            raise ValueError(f"Texture {texture_name} not found")
        # Free the texture unit for reuse
        cls.free_texture_unit(sampler.texture_unit)
        gl.glDeleteTextures(sampler.id)
        del cls._textures[sampler.uniform_name]

    @classmethod
    def get_texture_id(cls, texture_name: str):
        return cls._textures.get(texture_name).id

    @classmethod
    def get_texture_unit(cls, texture_name: str):
        return cls._textures.get(texture_name).texture_unit

    @classmethod
    def get_texture_resolution(cls, texture_name: str):
        return cls._textures.get(texture_name).resolution

    @classmethod
    def get_texture_channels(cls, texture_name: str):
        return cls._textures.get(texture_name).channels

    @classmethod
    def get_fbo_id(cls, fbo_name: str):
        return cls._fbo.get(fbo_name).id

    @classmethod
    def get_fbo_resolution(cls, fbo_name: str):
        return cls._fbo.get(fbo_name).resolution

    @classmethod
    def get_fbo_texture_name(cls, fbo_name: str):
        return cls._fbo.get(fbo_name).texture_name

    @classmethod
    def download_texture(cls, texture_name: str) -> Optional[torch.Tensor]:
        """
        Download a texture from OpenGL GPU memory to CPU as a torch tensor.

        Args:
            texture_name: Name of the texture to download

        Returns:
            Tensor of shape (C, H, W) or None if texture doesn't exist
        """
        sampler = cls._textures.get(texture_name)
        if sampler is None:
            print(f"Warning: Texture {texture_name} not found")
            return None

        # Get texture info
        texture_id = sampler.id
        texture_unit = sampler.texture_unit
        height, width = sampler.resolution
        channels = sampler.channels

        # Bind and download the texture from GPU
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        # Determine format based on channels
        if channels == 3:
            gl_format = gl.GL_RGB
        elif channels == 4:
            gl_format = gl.GL_RGBA
        else:
            print(
                f"Warning: Unsupported channel count {channels}. Only RGB (3) and RGBA (4) are supported."
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            return None

        # Read pixels from texture
        pixels = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl_format, gl.GL_FLOAT)

        # Convert to numpy array with correct shape (H, W, C)
        image_array = np.frombuffer(pixels, dtype=np.float32).reshape(
            height, width, channels
        )

        # Convert to torch tensor (C, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        check_for_errors()

        return image_tensor
