from PIL import Image
import torch
import numpy as np
import io


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch tensor (C, H, W) in [0, 1] range to PIL Image."""
    img_array = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(img_array, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to torch tensor (C, H, W) in [0, 1] range."""
    img_array = torch.from_numpy(np.array(image)).float() / 255.0
    return img_array.permute(2, 0, 1)


def encode_image_as_jpeg(image: Image.Image) -> bytes:
    """Encode a PIL Image as JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def expand_to_aspect_ratio(
    tensor: torch.Tensor,
    target_aspect_ratio: float,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> torch.Tensor:
    """Expand image to a target aspect ratio by padding with background color.

    Args:
        tensor: Input image tensor with shape (C, H, W) in [0, 1] range
        target_aspect_ratio: Target width/height ratio (e.g., 16/9 for 16:9)
        background_color: RGB background color as floats in [0, 1] range
        x_offset: Horizontal offset ratio from -1 to 1 (0 = centered, 1 = right, -1 = left)
        y_offset: Vertical offset ratio from -1 to 1 (0 = centered, 1 = down, -1 = up)

    Returns:
        Expanded tensor with shape (C, new_H, new_W) maintaining aspect ratio
    """
    c, h, w = tensor.shape
    current_aspect = w / h

    # If already at target aspect ratio, return original
    if abs(current_aspect - target_aspect_ratio) < 0.001:
        return tensor

    # Calculate new dimensions
    if current_aspect > target_aspect_ratio:
        # Image is too wide, add height
        new_w = w
        new_h = int(w / target_aspect_ratio)
    else:
        # Image is too tall, add width
        new_h = h
        new_w = int(h * target_aspect_ratio)

    # Create background canvas
    bg = torch.tensor(background_color, dtype=tensor.dtype, device=tensor.device)
    bg = bg.view(3, 1, 1).expand(c, new_h, new_w).clone()

    # Calculate position based on offset ratios
    # Available padding space
    padding_y = new_h - h
    padding_x = new_w - w

    # Calculate offset from centered position
    # offset ratio of 0 = centered, 1 = max positive, -1 = max negative
    # For centering: offset = padding / 2
    # With ratio: offset = padding / 2 + (padding / 2) * ratio
    offset_y = int(padding_y / 2 + (padding_y / 2) * (-y_offset))
    offset_x = int(padding_x / 2 + (padding_x / 2) * x_offset)

    # Place original image at calculated position
    bg[:, offset_y : offset_y + h, offset_x : offset_x + w] = tensor

    return bg


def resize_image(
    tensor: torch.Tensor,
    scale_percentage: float,
) -> torch.Tensor:
    """Resize image by a percentage scale factor.

    Args:
        tensor: Input image tensor with shape (C, H, W) in [0, 1] range
        scale_percentage: Scale factor as percentage (e.g., 50.0 for 50%, 200.0 for 200%)

    Returns:
        Resized tensor with shape (C, new_H, new_W)
    """
    # If scale is 100%, return original
    if abs(scale_percentage - 100.0) < 0.001:
        return tensor

    # Calculate scale factor
    scale = scale_percentage / 100.0

    c, h, w = tensor.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Ensure dimensions are at least 1 pixel
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Use torch.nn.functional.interpolate for high-quality resizing
    # Need to add batch dimension for interpolate
    tensor_batch = tensor.unsqueeze(0)  # (1, C, H, W)

    # Use bicubic interpolation for better quality
    resized = torch.nn.functional.interpolate(
        tensor_batch,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    # Remove batch dimension and clamp to valid range
    return resized.squeeze(0).clamp(0, 1)
