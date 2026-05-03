import re
import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F


def read_cube_file(lut_path: str) -> Tuple[torch.Tensor, list, list]:
    with open(lut_path, "r") as f:
        lines = f.readlines()

    # Parse header information
    lut_size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]

    # Find where the actual LUT data starts
    data_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip comments and empty lines
        if line.startswith("#") or not line:
            continue

        # Parse LUT_3D_SIZE
        if line.startswith("LUT_3D_SIZE"):
            lut_size = int(line.split()[1])

        # Parse DOMAIN_MIN (optional)
        elif line.startswith("DOMAIN_MIN"):
            domain_min = [float(x) for x in line.split()[1:4]]

        # Parse DOMAIN_MAX (optional)
        elif line.startswith("DOMAIN_MAX"):
            domain_max = [float(x) for x in line.split()[1:4]]

        # Check if this line looks like RGB data (3 floats)
        elif re.match(r"^[\d\.\-\s]+$", line) and len(line.split()) == 3:
            data_start_idx = i
            break

    if lut_size is None:
        raise ValueError("LUT_3D_SIZE not found in cube file")

    # Parse the RGB data
    lut_data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith("#"):
            try:
                r, g, b = map(float, line.split())
                lut_data.append([r, g, b])
            except ValueError:
                continue  # Skip invalid lines

    # Verify we have the right amount of data
    expected_entries = lut_size**3
    if len(lut_data) != expected_entries:
        raise ValueError(f"Expected {expected_entries} entries, got {len(lut_data)}")

    # Convert to numpy array and reshape with Fortran order
    lut_array = np.array(lut_data, dtype=np.float32)
    lut_cube_np = lut_array.reshape((lut_size, lut_size, lut_size, 3), order="F")

    # Create LUT instance
    return torch.from_numpy(lut_cube_np).contiguous(), domain_min, domain_max


def apply_lut(
    image: torch.Tensor,
    lut_tensor: torch.Tensor,
    domain_min: list = [0.0, 0.0, 0.0],
    domain_max: list = [1.0, 1.0, 1.0],
) -> torch.Tensor:
    is_batched = image.ndim == 4
    lut_tensor = lut_tensor.to(image)

    # Normalize to (B, H, W, C) format
    if is_batched:
        if image.shape[1] == 3:  # (B, C, H, W)
            x = image.permute(0, 2, 3, 1)
            channels_first = True
        else:  # (B, H, W, C)
            x = image
            channels_first = False
    else:
        # Add batch dimension for single image
        if image.shape[0] == 3:  # (C, H, W)
            x = image.permute(1, 2, 0).unsqueeze(0)
            channels_first = True
        else:  # (H, W, C)
            x = image.unsqueeze(0)
            channels_first = False

    B, H, W, C = x.shape

    # Check if LUT is grayscale (single-channel)
    is_grayscale_lut = lut_tensor.shape[-1] == 1
    lut_channels = lut_tensor.shape[-1]

    # Apply domain scaling
    domain_min_t = torch.tensor(domain_min, device=x.device)
    domain_max_t = torch.tensor(domain_max, device=x.device)
    domain_scaled = (x - domain_min_t) / (domain_max_t - domain_min_t)

    # Clamp coordinates for LUT lookup
    clamped_coords = torch.clamp(domain_scaled, 0, 1)

    # Prepare for grid_sample: need (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
    lut = lut_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, R, G, B)
    lut = lut.expand(B, -1, -1, -1, -1)  # (B, C, R, G, B)

    # Flip RGB to BGR for correct sampling
    clamped_coords_bgr = clamped_coords.flip(-1)

    # Scale from [0, 1] to [-1, 1] for grid_sample
    coords = clamped_coords_bgr * 2.0 - 1.0

    # Reshape coordinates to (B, H*W, 1, 1, 3) for sampling
    coords = coords.view(B, H * W, 1, 1, 3)

    # Sample the LUT with trilinear interpolation
    lut_sampled = F.grid_sample(
        lut, coords, mode="bilinear", padding_mode="border", align_corners=False
    )

    # Reshape LUT output back to (B, H, W, C)
    lut_sampled = lut_sampled.view(B, lut_channels, H, W).permute(0, 2, 3, 1)

    # If grayscale LUT, replicate single channel to 3 channels
    if is_grayscale_lut:
        result = lut_sampled.repeat(1, 1, 1, 3)  # (B, H, W, 3)
    else:
        result = lut_sampled

    # Return in original format
    if not is_batched:
        result = result.squeeze(0)  # Remove batch dimension
        if channels_first:
            return result.permute(2, 0, 1)
        else:
            return result
    else:
        if channels_first:
            return result.permute(0, 3, 1, 2)
        else:
            return result


def identity_lut(resolution: int = 32) -> torch.Tensor:
    """
    Create identity LUT using meshgrid.
    Uses BGR indexing order to match cube file format.
    At position [b,g,r], outputs RGB value [r,g,b] to preserve original color.
    """
    coords = torch.linspace(0, 1, resolution)
    r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([r, g, b], dim=-1)
