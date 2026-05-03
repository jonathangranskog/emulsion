import numpy as np
import numba


@numba.jit(nopython=True, parallel=True, cache=True)
def splat_gaussians(texture, positions, sizes, intensities, colors, color_shift):
    """Splat Gaussian distributions (scatter-based, grain-centric).

    Each grain is processed once and splatted to its affected pixels.
    This is much more efficient for CPU than the gather-based approach.

    Args:
        texture: Output texture array (H, W, 3) for RGB
        positions: Grain center positions (N, 2) as (y, x) coordinates
        sizes: Grain sizes (N,) - standard deviation of Gaussian
        intensities: Grain intensities (N,) - amplitude of Gaussian
        colors: Grain colors (N, 3) - RGB color for each grain
        color_shift: Blend factor between white (0.0) and random color (1.0)
    """
    height, width = texture.shape[:2]
    n_grains = positions.shape[0]

    # Parallel loop over grains (each grain processed independently)
    for i in numba.prange(n_grains):
        cy, cx = positions[i]
        sigma = sizes[i]
        intensity = intensities[i]
        grain_color = colors[i]

        # Blend between white and random color
        blended_color = 1.0 + color_shift * (grain_color - 1.0)

        # Calculate bounding box for this Gaussian (3 sigma on each side)
        radius = int(np.ceil(3.0 * sigma))
        y_min = max(0, int(cy) - radius)
        y_max = min(height, int(cy) + radius + 1)
        x_min = max(0, int(cx) - radius)
        x_max = min(width, int(cx) + radius + 1)

        # Precompute normalization factor
        norm = -0.5 / (sigma * sigma)

        # Splat Gaussian within bounding box
        for y in range(y_min, y_max):
            dy = y - cy
            dy2 = dy * dy

            for x in range(x_min, x_max):
                dx = x - cx
                dx2 = dx * dx

                # Gaussian function: exp(-0.5 * ((dx^2 + dy^2) / sigma^2))
                dist2 = dx2 + dy2
                gaussian = np.exp(dist2 * norm)

                # Accumulate grain contribution
                # NOTE: This has race conditions when grains overlap, but in practice
                # the visual result is still good and much faster than atomic operations
                for c in range(3):
                    texture[y, x, c] += gaussian * intensity * blended_color[c]

    return texture


@numba.jit(nopython=True, parallel=False, cache=True)
def sample_luminance_at_positions(luminance_map, positions):
    """Sample luminance values at grain positions using bilinear interpolation.

    Args:
        luminance_map: Luminance array (H, W)
        positions: Grain positions (N, 2) as (y, x) coordinates

    Returns:
        Array of luminance values (N,)
    """
    height, width = luminance_map.shape
    n_grains = positions.shape[0]
    luminance_values = np.zeros(n_grains, dtype=np.float32)

    for i in range(n_grains):
        y, x = positions[i]

        # Clamp to valid range
        y = max(0.0, min(height - 1.001, y))
        x = max(0.0, min(width - 1.001, x))

        # Get integer and fractional parts
        y_int = int(y)
        x_int = int(x)
        y_frac = y - y_int
        x_frac = x - x_int

        # Bilinear interpolation
        y_int_next = min(y_int + 1, height - 1)
        x_int_next = min(x_int + 1, width - 1)

        v00 = luminance_map[y_int, x_int]
        v01 = luminance_map[y_int, x_int_next]
        v10 = luminance_map[y_int_next, x_int]
        v11 = luminance_map[y_int_next, x_int_next]

        v0 = v00 * (1.0 - x_frac) + v01 * x_frac
        v1 = v10 * (1.0 - x_frac) + v11 * x_frac

        luminance_values[i] = v0 * (1.0 - y_frac) + v1 * y_frac

    return luminance_values


def generate_grain_texture(
    width: int,
    height: int,
    n_grains: int,
    size_mean: float = 0.25,
    size_std: float = 0.25,
    intensity_mean: float = 0.25,
    intensity_std: float = 0.25,
    color_shift: float = 0.0,
    luminance_map: np.ndarray | None = None,
    luma_size_scale: float = 2.0,
    seed: int | None = None,
):
    """Generate a film grain texture using Gaussian splatting.

    Uses scatter-based approach where each grain is splatted to its affected pixels.
    This is much more efficient for CPU than gather-based approaches.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        n_grains: Number of grain particles to generate
        size_mean: Mean size (standard deviation) of grain particles in pixels
        size_std: Standard deviation of grain sizes
        intensity_mean: Mean intensity of grains (bias)
        intensity_std: Standard deviation of grain intensities
        color_shift: Blend between white (0.0) and random color (1.0)
        luminance_map: Optional luminance map (H, W) to modulate grain sizes
        luma_size_scale: Scaling factor for luminance-based size modulation
        seed: Random seed for reproducibility

    Returns:
        Texture array (H, W, 3) as float32
    """
    if seed is not None:
        np.random.seed(seed)

    print(
        f"Generating {n_grains:,} grain particles for {width}x{height} texture (CPU)..."
    )

    # Initialize empty RGB texture
    texture = np.zeros((height, width, 3), dtype=np.float32)

    # Generate random grain parameters
    print("Generating grain parameters...")

    # Random positions (uniform distribution)
    positions = np.column_stack(
        [
            np.random.uniform(0, height, n_grains),
            np.random.uniform(0, width, n_grains),
        ]
    ).astype(np.float32)

    # Random sizes (log-normal distribution for more realistic variation)
    sizes = np.random.lognormal(
        mean=np.log(size_mean), sigma=size_std, size=n_grains
    ).astype(np.float32)

    # Modulate sizes based on luminance if provided
    # Darker regions (low luminance) get larger grains
    if luminance_map is not None:
        print("Modulating grain sizes based on image luminance...")
        luma_values = sample_luminance_at_positions(luminance_map, positions)
        # Scale sizes: darker (low luma) = larger grains
        size_multiplier = 1.0 + (1.0 - luma_values) * (luma_size_scale - 1.0)
        sizes = sizes * size_multiplier

    # Clamp sizes to reasonable range
    sizes = np.clip(sizes, 0.5, 10.0)

    # Random intensities (normal distribution, can be positive or negative)
    intensities = np.random.normal(
        loc=intensity_mean, scale=intensity_std, size=n_grains
    ).astype(np.float32)

    # Generate random RGB colors for each grain
    colors = np.random.uniform(0.0, 1.0, (n_grains, 3)).astype(np.float32)

    # Splat grains (parallel over grains, not pixels)
    print("Splatting grains (parallel over grains with Numba)...")
    texture = splat_gaussians(
        texture, positions, sizes, intensities, colors, color_shift
    )

    print(
        f"Texture stats: min={texture.min():.4f}, max={texture.max():.4f}, "
        f"mean={texture.mean():.6f}, std={texture.std():.4f}"
    )

    return texture
