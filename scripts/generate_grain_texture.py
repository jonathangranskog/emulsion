#!/usr/bin/env python3
"""Generate film grain textures using Gaussian splatting.

This script generates realistic film grain textures by splatting many small
Gaussian distributions across a texture. This creates more irregular and
interesting patterns than procedural noise functions like simplex noise.

The Gaussian splatting approach better simulates the random distribution of
silver halide crystals in traditional film emulsions.
"""

import numpy as np
import numba
from PIL import Image
import argparse
from pathlib import Path

from src.grain.gaussian_splatting import generate_grain_texture


def load_image_as_luminance(image_path, target_width=None, target_height=None):
    """Load an image and convert to normalized luminance map.

    Args:
        image_path: Path to input image
        target_width: If specified, resize to this width
        target_height: If specified, resize to this height

    Returns:
        Luminance array of shape (height, width) with values in [0, 1]
    """
    img = Image.open(image_path)

    # Resize if target dimensions specified
    if target_width and target_height:
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Calculate luminance using Rec. 709 weights
    luminance = (
        0.2126 * img_array[:, :, 0]
        + 0.7152 * img_array[:, :, 1]
        + 0.0722 * img_array[:, :, 2]
    )

    return luminance


def save_texture(texture, output_path, normalize=True):
    """Save texture to an image file.

    Args:
        texture: Grain texture array - can be (H, W) or (H, W, 3)
        output_path: Output file path
        normalize: If True, normalize to [0, 1] range before saving
    """
    if normalize:
        # Normalize to [0, 1] range
        texture_min = texture.min()
        texture_max = texture.max()

        if texture_max > texture_min:
            texture_normalized = (texture - texture_min) / (texture_max - texture_min)
        else:
            texture_normalized = np.zeros_like(texture)
    else:
        # Assume already in [0, 1] range, just clamp
        texture_normalized = np.clip(texture, 0.0, 1.0)

    # Check if grayscale or RGB
    if len(texture.shape) == 2:
        # Grayscale - save as 16-bit
        texture_16bit = (texture_normalized * 65535).astype(np.uint16)
        img = Image.fromarray(texture_16bit, mode="I;16")
    else:
        # RGB - save as 8-bit RGB
        texture_8bit = (texture_normalized * 255).astype(np.uint8)
        img = Image.fromarray(texture_8bit, mode="RGB")

    img.save(output_path)
    print(f"Saved grain texture to {output_path}")


def create_composite(image_path, grain_texture, output_path):
    """Create composite image by adding grain to input image.

    Args:
        image_path: Path to input image
        grain_texture: Grain texture array (H, W, 3) RGB
        output_path: Path to save composite image
    """
    # Load input image
    img = Image.open(image_path)

    # Get dimensions
    height, width = grain_texture.shape[:2]

    # Resize image to match grain texture if needed
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add grain to image (both are RGB)
    composite = img_array + grain_texture

    # Clamp to valid range
    composite = np.clip(composite, 0.0, 1.0)

    # Convert back to 8-bit and save
    composite_8bit = (composite * 255).astype(np.uint8)
    composite_img = Image.fromarray(composite_8bit, mode="RGB")
    composite_img.save(output_path)
    print(f"Saved composite image to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate film grain textures using Gaussian splatting"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4096,
        help="Texture width in pixels (default: 4096)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=4096,
        help="Texture height in pixels (default: 4096)",
    )
    parser.add_argument(
        "--n-grains",
        type=int,
        default=500000,
        help="Number of grain particles (default: 500000)",
    )
    parser.add_argument(
        "--size-mean",
        type=float,
        default=1.5,
        help="Mean grain size in pixels (default: 1.5)",
    )
    parser.add_argument(
        "--size-std",
        type=float,
        default=0.5,
        help="Standard deviation of grain sizes (default: 0.5)",
    )
    parser.add_argument(
        "--intensity-mean",
        type=float,
        default=0.0,
        help="Mean grain intensity (default: 0.0)",
    )
    parser.add_argument(
        "--intensity-std",
        type=float,
        default=0.02,
        help="Standard deviation of grain intensities (default: 0.02)",
    )
    parser.add_argument(
        "--color-shift",
        type=float,
        default=0.0,
        help="Blend between white (0.0) and random RGB colors (1.0) (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grain_texture.png",
        help="Output file path (default: grain_texture.png)",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Input image to use for luminance-based grain size modulation",
    )
    parser.add_argument(
        "--luma-size-scale",
        type=float,
        default=2.0,
        help="Luminance-based size scaling factor - larger grains in dark regions (default: 2.0)",
    )
    parser.add_argument(
        "--composite-output",
        type=str,
        default=None,
        help="Output file path for composite image (default: auto-generated from output path)",
    )

    args = parser.parse_args()

    # Load input image if provided
    luminance_map = None
    if args.input_image:
        print(f"Loading input image: {args.input_image}")
        luminance_map = load_image_as_luminance(
            args.input_image, target_width=args.width, target_height=args.height
        )
        print(
            f"Luminance map loaded: {luminance_map.shape}, "
            f"range [{luminance_map.min():.3f}, {luminance_map.max():.3f}]"
        )

    # Generate texture
    texture = generate_grain_texture(
        width=args.width,
        height=args.height,
        n_grains=args.n_grains,
        size_mean=args.size_mean,
        size_std=args.size_std,
        intensity_mean=args.intensity_mean,
        intensity_std=args.intensity_std,
        color_shift=args.color_shift,
        luminance_map=luminance_map,
        luma_size_scale=args.luma_size_scale,
        seed=args.seed,
    )

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save grain texture
    save_texture(texture, output_path, normalize=False)

    # Create composite image if input image was provided
    if args.input_image:
        if args.composite_output:
            composite_output_path = Path(args.composite_output)
        else:
            # Auto-generate composite output path
            composite_output_path = output_path.with_stem(
                output_path.stem + "_composite"
            )

        composite_output_path.parent.mkdir(parents=True, exist_ok=True)
        create_composite(args.input_image, texture, composite_output_path)

    print("Done!")


if __name__ == "__main__":
    main()
