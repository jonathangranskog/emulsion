import os
from pathlib import Path
from typing import Optional

import exifread
import numpy as np
import rawpy
import torch
import torch.nn.functional as F

from PIL import Image
from src.core.image import ImageData
from src.core.metadata import ImageMetadata
from src.effects.temperature_tint import kelvin_to_rgb

# Supported image formats
RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


class ImageLoader:
    def __init__(self):
        pass

    @staticmethod
    def apply_sensor_crop(
        rgb: torch.Tensor, sizes: rawpy.ImageSizes | None
    ) -> torch.Tensor:
        if sizes is None:
            return rgb

        # These are in the sensor coordinates
        crop_left = sizes.crop_left_margin
        crop_top = sizes.crop_top_margin
        crop_width = sizes.crop_width
        crop_height = sizes.crop_height

        # Apply the crop
        return rgb[
            :, crop_top : crop_top + crop_height, crop_left : crop_left + crop_width
        ]

    @staticmethod
    def _extract_raw_white_balance(raw: rawpy.RawPy) -> dict:
        """Extract white balance and color matrix metadata from a RAW file.

        Returns a dict with keys prefixed by 'raw_' containing:
          - raw_camera_wb: camera white balance multipliers [R, G, B, G2]
          - raw_daylight_wb: daylight white balance multipliers [R, G, B, G2]
          - raw_as_shot_kelvin: estimated as-shot color temperature in Kelvin

        The CCT is estimated by comparing the camera and daylight white
        balance R/B gain ratios and finding the Planckian locus temperature
        that matches.
        """
        result = {}
        try:
            camera_wb = list(raw.camera_whitebalance)
            daylight_wb = list(raw.daylight_whitebalance)

            result["raw_camera_wb"] = camera_wb
            result["raw_daylight_wb"] = daylight_wb

            # Estimate as-shot CCT from the ratio of camera to daylight
            # WB multipliers.  The R/B ratio of the gains encodes the
            # scene colour temperature relative to the daylight reference.
            cam_r, cam_b = camera_wb[0], camera_wb[2]
            day_r, day_b = daylight_wb[0], daylight_wb[2]

            if all(v > 0 for v in (cam_r, cam_b, day_r, day_b)):
                # Ratio of camera-to-daylight R/B gains.  A higher R/B in
                # the camera WB means the scene was bluer (higher Kelvin),
                # because the sensor needs more red gain to compensate.
                q = (cam_r / cam_b) / (day_r / day_b)

                # The daylight WB reference is approximately 5 500 K.
                daylight_kelvin = 5500.0
                ref_r, _, ref_b = kelvin_to_rgb(daylight_kelvin)
                # Target Planckian B/R ratio for the as-shot illuminant
                target_br = (ref_b / max(ref_r, 1e-10)) * q

                # Binary search along the Planckian locus for the matching
                # temperature.  B/R increases monotonically with Kelvin.
                lo, hi = 1500.0, 25000.0
                for _ in range(50):
                    mid = (lo + hi) / 2.0
                    r, _, b = kelvin_to_rgb(mid)
                    br = b / max(r, 1e-10)
                    if br < target_br:
                        lo = mid
                    else:
                        hi = mid
                cct = (lo + hi) / 2.0
                cct = max(1500.0, min(25000.0, cct))
                result["raw_as_shot_kelvin"] = float(cct)
        except Exception as e:
            print(f"Could not extract RAW white balance metadata: {e}")

        return result

    @staticmethod
    def load_image(path: str) -> ImageData:
        assert os.path.exists(path), f"The image file {path} does not exist."
        with rawpy.imread(path) as raw:
            # Read the sizes
            if hasattr(raw, "sizes"):
                sizes = raw.sizes
            else:
                sizes = None

            # Attempt to extract the thumbnail
            thumb = ImageLoader.attempt_extract_thumbnail(raw)

            # Extract RAW white balance metadata before postprocessing
            raw_wb_metadata = ImageLoader._extract_raw_white_balance(raw)

            # Postprocess the image at the end
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0,
            )

        # Read the EXIF data
        with open(path, "rb") as f:
            tags = exifread.process_file(f)

        # Merge RAW white balance metadata into EXIF tags
        tags.update(raw_wb_metadata)
        ImageMetadata.set(tags)

        # Perform additional cropping if included
        # TODO: Add these as effects to the image stack instead
        if rgb.dtype == np.uint16:
            tensor = rgb.astype(np.float32) / 65535.0
        else:
            tensor = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).contiguous()
        tensor = ImageLoader.apply_sensor_crop(tensor, sizes)
        tensor = ImageLoader.apply_digital_zoom(tensor, tags)
        tensor = ImageLoader.apply_flip(tensor, tags)
        tensor = ImageLoader.apply_black_and_white(tensor, thumb, tags)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return ImageData(tensor=tensor, metadata=tags)

    @staticmethod
    def apply_digital_zoom(rgb: torch.Tensor, tags: dict) -> torch.Tensor:
        if "EXIF DigitalZoomRatio" not in tags:
            return rgb
        zoom_ratio = float(tags["EXIF DigitalZoomRatio"].values[0])
        if zoom_ratio > 1:
            h, w = rgb.shape[1:]
            new_h = int(h / zoom_ratio)
            new_w = int(w / zoom_ratio)
            crop_left = (w - new_w) // 2
            crop_top = (h - new_h) // 2

            # Force dimensions to be multiples of 4
            # For some reason, not doing this creates mosaicing patterns
            new_h = (new_h // 4) * 4
            new_w = (new_w // 4) * 4
            crop_left = ((w - new_w) // 2 // 4) * 4
            crop_top = ((h - new_h) // 2 // 4) * 4

            # Create a new tensor instead of slicing
            cropped = torch.zeros(
                (rgb.shape[0], new_h, new_w), dtype=rgb.dtype, device=rgb.device
            )
            cropped.copy_(
                rgb[:, crop_top : crop_top + new_h, crop_left : crop_left + new_w]
            )
            return cropped
        else:
            return rgb

    @staticmethod
    def apply_flip(rgb: torch.Tensor, tags: dict) -> torch.Tensor:
        if "Image Orientation" not in tags:
            return rgb
        orientation = int(tags["Image Orientation"].values[0])
        if orientation == 1:
            # Normal (0° rotation) - no change
            pass
        elif orientation == 2:
            # Mirrored horizontally
            rgb = rgb.flip(2)  # flip along width dimension
        elif orientation == 3:
            # Rotated 180°
            rgb = rgb.flip(1).flip(2)  # flip both height and width
        elif orientation == 4:
            # Mirrored vertically
            rgb = rgb.flip(1)  # flip along height dimension
        elif orientation == 5:
            # Mirrored horizontally and rotated 270° clockwise
            rgb = rgb.flip(2).permute(0, 2, 1).flip(1)
        elif orientation == 6:
            # Rotated 90° clockwise
            rgb = rgb.permute(0, 2, 1).flip(2)  # transpose then flip horizontally
        elif orientation == 7:
            # Mirrored horizontally and rotated 90° clockwise
            rgb = rgb.flip(2).permute(0, 2, 1).flip(2)
        elif orientation == 8:
            # Rotated 270° clockwise
            rgb = rgb.permute(0, 2, 1).flip(1)  # transpose then flip vertically

        return rgb

    @staticmethod
    def attempt_extract_thumbnail(raw: rawpy.RawPy) -> torch.Tensor | None:
        try:
            # Extract the thumbnail
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                # It's a JPEG thumbnail
                from io import BytesIO

                thumb_image = Image.open(BytesIO(thumb.data))
                thumb_array = np.array(thumb_image)
                thumb_tensor = (
                    torch.from_numpy(thumb_array).permute(2, 0, 1).contiguous()
                )
                return thumb_tensor
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                thumb_array = np.array(thumb.data)
                thumb_tensor = (
                    torch.from_numpy(thumb_array).permute(2, 0, 1).contiguous()
                )
                return thumb_tensor
        except rawpy.LibRawError as e:
            print(f"Could not extract thumbnail: {e}")
            return None
        return None

    @staticmethod
    def apply_black_and_white(
        rgb: torch.Tensor, thumb: torch.Tensor | None, tags: dict
    ) -> torch.Tensor:
        """
        Convert to grayscale if the image is intended to be monochrome.
        If the thumbnail is grayscale, we can make the image grayscale as well.
        """

        def make_grayscale(rgb: torch.Tensor) -> torch.Tensor:
            weights = torch.tensor(
                [0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype
            )
            grayscale = (
                (rgb * weights.view(3, 1, 1)).sum(dim=0, keepdim=True).repeat(3, 1, 1)
            )
            return grayscale

        if thumb is not None and thumb.shape[0] == 1:
            # Convert RGB to grayscale using luminance weights
            # Using ITU-R BT.709 coefficients for better perceptual accuracy
            return make_grayscale(rgb)
        elif thumb is not None and thumb.shape[0] == 3:
            r, g, b = thumb[0, :, :], thumb[1, :, :], thumb[2, :, :]
            # Check if channels are nearly identical
            rg_diff = torch.abs(r.float() - g.float()).mean()
            rb_diff = torch.abs(r.float() - b.float()).mean()
            if rg_diff < 2.0 and rb_diff < 2.0:
                return make_grayscale(rgb)
        return rgb

    @staticmethod
    def is_supported_image(path: str) -> bool:
        """Check if a file is a supported image format."""
        ext = Path(path).suffix.lower()
        return ext in RAW_EXTENSIONS or ext in STANDARD_EXTENSIONS

    @staticmethod
    def load_standard_image(path: str) -> ImageData:
        """Load a standard image format (JPG, PNG, etc.) without RAW processing."""
        img = Image.open(path).convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        ImageMetadata.set({"path": path})
        return ImageData(tensor=tensor.contiguous(), metadata={"path": path})

    @staticmethod
    def load_image_safe(path: str) -> Optional[ImageData]:
        """Load an image, returning None on failure instead of raising."""
        try:
            ext = Path(path).suffix.lower()
            if ext in RAW_EXTENSIONS:
                return ImageLoader.load_image(path)
            else:
                return ImageLoader.load_standard_image(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None

    @staticmethod
    def resize_for_preview(tensor: torch.Tensor, max_size: int = 1024) -> torch.Tensor:
        """Resize tensor to fit within max_size while preserving aspect ratio."""
        _, h, w = tensor.shape
        if max(h, w) <= max_size:
            return tensor

        scale = max_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return resized.contiguous()
