"""
Interactive crop tool for the image viewer.

Provides visual crop overlay with handles, aspect ratio constraints,
and keyboard shortcuts for intuitive image cropping.
"""

import math
import glfw
import glm
from imgui_bundle import imgui
import imgui.integrations.glfw  # Required for proper imgui context

from src.effects.crop import Crop


class CropTool:
    """Interactive crop tool with visual overlay and aspect ratio constraints."""

    def __init__(self, viewer):
        """
        Initialize the crop tool.

        Args:
            viewer: The Viewer instance (provides access to effects, window, etc.)
        """
        self.viewer = viewer

        # Crop mode state
        self.active = False
        self.rect = None  # {"x": float, "y": float, "width": float, "height": float}
        self.dragging_handle = None
        self.dragging_rect = False
        self.drag_start_mouse = None
        self.drag_start_rect = None

        # Aspect ratio options
        self.aspect_ratios = [
            ("Free", None),
            ("1:1 (Square)", 1.0),
            ("5:4", 5.0 / 4.0),
            ("4:3", 4.0 / 3.0),
            ("7:6", 7.0 / 6.0),
            ("3:2", 3.0 / 2.0),
            ("16:9", 16.0 / 9.0),
            ("21:9 (Ultrawide)", 21.0 / 9.0),
            ("65:24 (XPan)", 65.0 / 24.0),
            ("4:5 (Portrait)", 4.0 / 5.0),
            ("5:7 (Portrait)", 5.0 / 7.0),
            ("6:7 (Portrait)", 6.0 / 7.0),
            ("2:3 (Portrait)", 2.0 / 3.0),
            ("9:16 (Vertical)", 9.0 / 16.0),
        ]
        self.aspect_ratio_idx = 0  # Default to "Free"

    def activate(self):
        """Activate crop mode and initialize crop rect"""
        self.active = True

        # Get source image dimensions
        source_tensor = self.viewer.effects_processor.source.tensor
        _, source_height, source_width = source_tensor.shape

        # Set viewer's image dimensions to source dimensions for crop mode
        self.viewer.image_width = source_width
        self.viewer.image_height = source_height

        # Check if there's already a Crop effect at the beginning of the stack
        existing_crop = None
        if len(self.viewer.effects_stack.effects) > 0 and isinstance(
            self.viewer.effects_stack.effects[0], Crop
        ):
            existing_crop = self.viewer.effects_stack.effects[0]

        # If there's an existing crop, use its parameters (with validation)
        if existing_crop:
            safe_width = min(existing_crop.width, source_width)
            safe_height = min(existing_crop.height, source_height)
            safe_x = max(0, min(existing_crop.x, source_width - safe_width))
            safe_y = max(0, min(existing_crop.y, source_height - safe_height))

            self.rect = {
                "x": safe_x,
                "y": safe_y,
                "width": safe_width,
                "height": safe_height,
            }
        else:
            # Initialize crop rect to full image (no margin)
            self.rect = {
                "x": 0,
                "y": 0,
                "width": source_width,
                "height": source_height,
            }

        # Apply current aspect ratio if one is selected
        _, current_aspect_ratio = self.aspect_ratios[self.aspect_ratio_idx]
        if current_aspect_ratio is not None:
            self.apply_aspect_ratio(current_aspect_ratio)

        # Validate the crop rect to ensure it's within bounds
        self.validate_rect()

    def cancel(self):
        """Cancel crop mode without applying"""
        self.active = False
        self.rect = None
        self.dragging_handle = None
        self.dragging_rect = False

    def apply(self):
        """Apply crop and exit crop mode"""
        if not self.rect:
            self.cancel()
            return

        # Get source dimensions
        source_tensor = self.viewer.effects_processor.source.tensor
        _, source_height, source_width = source_tensor.shape

        # Check if there's already a Crop effect at the beginning of the stack
        existing_crop = None
        if len(self.viewer.effects_stack.effects) > 0 and isinstance(
            self.viewer.effects_stack.effects[0], Crop
        ):
            existing_crop = self.viewer.effects_stack.effects[0]

        # Validate and clamp crop parameters
        new_x = max(0, min(int(self.rect["x"]), source_width - 1))
        new_y = max(0, min(int(self.rect["y"]), source_height - 1))
        new_width = max(1, min(int(self.rect["width"]), source_width - new_x))
        new_height = max(1, min(int(self.rect["height"]), source_height - new_y))

        if (
            new_x == 0
            and new_y == 0
            and new_width == source_width
            and new_height == source_height
        ):
            # If there's an existing crop and we're resetting to full image, remove it
            if existing_crop:
                self.viewer.effects_stack.effects.pop(0)
                self.viewer.effects_stack.mark_reconstruction_required()
                self.viewer.effects_stack.capture_state("Remove crop")
        elif existing_crop:
            # Update existing crop parameters
            existing_crop.x = new_x
            existing_crop.y = new_y
            existing_crop.width = new_width
            existing_crop.height = new_height
            existing_crop.source_width = source_width
            existing_crop.source_height = source_height

            self.viewer.effects_stack.mark_reconstruction_required()
            self.viewer.effects_stack.capture_state("Update crop")
        else:
            # Create new crop effect
            crop_effect = Crop(
                x=new_x,
                y=new_y,
                width=new_width,
                height=new_height,
                source_width=source_width,
                source_height=source_height,
            )

            # Insert at beginning of effect stack
            self.viewer.effects_stack.effects.insert(0, crop_effect)
            self.viewer.effects_stack.mark_reconstruction_required()
            self.viewer.effects_stack.capture_state("Add crop")

        self.active = False
        self.rect = None

    def screen_to_image_coords(self, screen_pos):
        """Convert screen coordinates to image coordinates"""
        # Get window size (ImGui uses window coords, not framebuffer)
        win_width, win_height = glfw.get_window_size(self.viewer.window)

        # Use current image dimensions (which will be source dimensions in crop mode)
        img_width, img_height = self.viewer.image_width, self.viewer.image_height

        # Create MVP matrices (use framebuffer size for rendering)
        fb_width, fb_height = glfw.get_framebuffer_size(self.viewer.window)
        projection, view, model = self.viewer.create_mvp(fb_height, fb_width)
        mvp = projection * view * model

        # Convert screen to NDC (using window size since mouse is in window coords)
        ndc_x = (2.0 * screen_pos.x / win_width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_pos.y / win_height)

        # Inverse MVP to get world coordinates
        try:
            inv_mvp = glm.inverse(mvp)
        except:
            return None

        world_pos = inv_mvp * glm.vec4(ndc_x, ndc_y, 0.0, 1.0)

        # World coordinates are in [-1, 1] range (aspect ratio already applied by MVP)
        # Convert directly to image coordinates
        if img_width > 0 and img_height > 0:
            img_x = (world_pos.x + 1.0) * 0.5 * img_width
            img_y = (1.0 - world_pos.y) * 0.5 * img_height
            return (img_x, img_y)

        return None

    def image_to_screen_coords(self, img_pos):
        """Convert image coordinates to screen coordinates"""
        # Get window size (ImGui draws in window coords, not framebuffer)
        win_width, win_height = glfw.get_window_size(self.viewer.window)

        # Use current image dimensions (which will be source dimensions in crop mode)
        img_width, img_height = self.viewer.image_width, self.viewer.image_height

        # Convert image coords to normalized [0, 1]
        norm_x = img_pos[0] / img_width if img_width > 0 else 0
        norm_y = img_pos[1] / img_height if img_height > 0 else 0

        # Convert to [-1, 1] world space (without aspect ratio - MVP handles that)
        world_x = norm_x * 2.0 - 1.0
        world_y = 1.0 - (norm_y * 2.0)

        # Apply MVP (use framebuffer size for rendering matrices)
        fb_width, fb_height = glfw.get_framebuffer_size(self.viewer.window)
        projection, view, model = self.viewer.create_mvp(fb_height, fb_width)
        mvp = projection * view * model
        screen_pos = mvp * glm.vec4(world_x, world_y, 0.0, 1.0)

        # Convert from NDC to window coords (not framebuffer)
        screen_x = (screen_pos.x + 1.0) * 0.5 * win_width
        screen_y = (1.0 - screen_pos.y) * 0.5 * win_height

        return (screen_x, screen_y)

    def handle_input(self, io):
        """Handle mouse input for crop mode"""
        if not self.rect:
            return

        # Don't handle crop input if ImGui wants the mouse
        if io.want_capture_mouse:
            if self.dragging_handle or self.dragging_rect:
                self.dragging_handle = None
                self.dragging_rect = False
            return

        mouse_pos = glm.vec2(io.mouse_pos.x, io.mouse_pos.y)

        # Convert mouse to image coordinates
        img_pos = self.screen_to_image_coords(mouse_pos)
        if img_pos is None:
            return

        # Mouse down - start drag
        if io.mouse_down[0]:
            if not self.dragging_handle and not self.dragging_rect:
                # Check if clicking on handle
                handle = self.get_handle_at(img_pos)
                if handle:
                    self.dragging_handle = handle
                    self.drag_start_mouse = img_pos
                    self.drag_start_rect = self.rect.copy()
                # Check if clicking inside rect
                elif self.is_point_in_rect(img_pos):
                    self.dragging_rect = True
                    self.drag_start_mouse = img_pos
                    self.drag_start_rect = self.rect.copy()

        # Mouse up - stop drag
        if not io.mouse_down[0]:
            self.dragging_handle = None
            self.dragging_rect = False

        # Update rect during drag
        if self.dragging_handle and self.drag_start_mouse:
            delta_x = img_pos[0] - self.drag_start_mouse[0]
            delta_y = img_pos[1] - self.drag_start_mouse[1]
            self.update_rect_from_handle(self.dragging_handle, delta_x, delta_y)
            self.validate_rect()

        elif self.dragging_rect and self.drag_start_mouse:
            delta_x = img_pos[0] - self.drag_start_mouse[0]
            delta_y = img_pos[1] - self.drag_start_mouse[1]
            self.move_rect(delta_x, delta_y)
            self.validate_rect()

    def validate_rect(self):
        """Ensure crop rect has valid dimensions within image bounds"""
        if not self.rect:
            return

        img_width = self.viewer.image_width
        img_height = self.viewer.image_height

        # Ensure all values are valid finite numbers
        for key in ["x", "y", "width", "height"]:
            value = self.rect[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                if key in ["width", "height"]:
                    self.rect[key] = min(
                        100, img_width if key == "width" else img_height
                    )
                else:
                    self.rect[key] = 0
                continue

        # Clamp dimensions to valid range
        self.rect["width"] = max(10, min(float(self.rect["width"]), img_width))
        self.rect["height"] = max(10, min(float(self.rect["height"]), img_height))

        # Clamp position to valid range
        self.rect["x"] = max(
            0, min(float(self.rect["x"]), img_width - self.rect["width"])
        )
        self.rect["y"] = max(
            0, min(float(self.rect["y"]), img_height - self.rect["height"])
        )

    def get_handle_at(self, img_pos):
        """Check if image position is near a crop handle"""
        if not self.rect:
            return None

        x, y = img_pos
        r = self.rect

        # Use larger threshold for easier handle grabbing
        img_width = self.viewer.image_width
        img_height = self.viewer.image_height
        threshold = max(50, min(img_width, img_height) * 0.02)

        handles = {
            "tl": (r["x"], r["y"]),
            "tr": (r["x"] + r["width"], r["y"]),
            "bl": (r["x"], r["y"] + r["height"]),
            "br": (r["x"] + r["width"], r["y"] + r["height"]),
            "t": (r["x"] + r["width"] / 2, r["y"]),
            "b": (r["x"] + r["width"] / 2, r["y"] + r["height"]),
            "l": (r["x"], r["y"] + r["height"] / 2),
            "r": (r["x"] + r["width"], r["y"] + r["height"] / 2),
        }

        for handle_name, (hx, hy) in handles.items():
            dist = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
            if dist < threshold:
                return handle_name

        return None

    def is_point_in_rect(self, img_pos):
        """Check if image position is inside crop rect"""
        if not self.rect:
            return False
        x, y = img_pos
        r = self.rect
        return (
            r["x"] <= x <= r["x"] + r["width"] and r["y"] <= y <= r["y"] + r["height"]
        )

    def update_rect_from_handle(self, handle, delta_x, delta_y):
        """Update crop rect based on handle drag"""
        if not self.drag_start_rect:
            return

        # Extract current rect values
        x = self.drag_start_rect["x"]
        y = self.drag_start_rect["y"]
        width = self.drag_start_rect["width"]
        height = self.drag_start_rect["height"]

        img_width = self.viewer.image_width
        img_height = self.viewer.image_height

        # Get current aspect ratio setting
        _, aspect_ratio = self.aspect_ratios[self.aspect_ratio_idx]

        # Update rect based on which handle is being dragged
        if aspect_ratio is None:
            # Free aspect ratio - original behavior
            if "t" in handle:
                new_y = max(0, min(y + delta_y, y + height - 10))
                height = height - (new_y - y)
                y = new_y
            if "b" in handle:
                height = max(10, min(height + delta_y, img_height - y))
            if "l" in handle:
                new_x = max(0, min(x + delta_x, x + width - 10))
                width = width - (new_x - x)
                x = new_x
            if "r" in handle:
                width = max(10, min(width + delta_x, img_width - x))
        else:
            # Locked aspect ratio
            if handle in ["tl", "tr", "bl", "br"]:
                # Corner handles
                if abs(delta_x) > abs(delta_y):
                    # Width is primary
                    if "l" in handle:
                        new_x = max(0, min(x + delta_x, x + width - 10))
                        width = width - (new_x - x)
                        x = new_x
                    else:
                        width = max(10, min(width + delta_x, img_width - x))
                    new_height = width / aspect_ratio
                    if "t" in handle:
                        y = y + height - new_height
                    height = new_height
                else:
                    # Height is primary
                    if "t" in handle:
                        new_y = max(0, min(y + delta_y, y + height - 10))
                        height = height - (new_y - y)
                        y = new_y
                    else:
                        height = max(10, min(height + delta_y, img_height - y))
                    new_width = height * aspect_ratio
                    if "l" in handle:
                        x = x + width - new_width
                    width = new_width
            else:
                # Edge handles
                if handle in ["l", "r"]:
                    if handle == "l":
                        new_x = max(0, min(x + delta_x, x + width - 10))
                        width = width - (new_x - x)
                        x = new_x
                    else:
                        width = max(10, min(width + delta_x, img_width - x))
                    new_height = width / aspect_ratio
                    y = y + (height - new_height) / 2
                    height = new_height
                else:  # handle in ["t", "b"]
                    if handle == "t":
                        new_y = max(0, min(y + delta_y, y + height - 10))
                        height = height - (new_y - y)
                        y = new_y
                    else:
                        height = max(10, min(height + delta_y, img_height - y))
                    new_width = height * aspect_ratio
                    x = x + (width - new_width) / 2
                    width = new_width

            # Clamp to image bounds
            width = max(10, min(width, img_width - x))
            height = max(10, min(height, img_height - y))
            x = max(0, min(x, img_width - width))
            y = max(0, min(y, img_height - height))

        # Update rect
        self.rect = {"x": x, "y": y, "width": width, "height": height}

    def move_rect(self, delta_x, delta_y):
        """Move entire crop rect"""
        if not self.drag_start_rect:
            return

        r = self.drag_start_rect.copy()
        img_width = self.viewer.image_width
        img_height = self.viewer.image_height
        r["x"] = max(0, min(r["x"] + delta_x, img_width - r["width"]))
        r["y"] = max(0, min(r["y"] + delta_y, img_height - r["height"]))
        self.rect = r

    def apply_aspect_ratio(self, aspect_ratio):
        """Apply aspect ratio constraint to current crop rect, maximizing to fill image"""
        if not self.rect:
            return

        r = self.rect
        img_width = self.viewer.image_width
        img_height = self.viewer.image_height

        # Maximize crop size while maintaining aspect ratio
        new_width = img_width
        new_height = new_width / aspect_ratio

        # If height exceeds image, constrain by height instead
        if new_height > img_height:
            new_height = img_height
            new_width = new_height * aspect_ratio

        # Ensure minimum size
        new_width = max(10, new_width)
        new_height = max(10, new_height)

        # Center the crop rect
        new_x = (img_width - new_width) / 2
        new_y = (img_height - new_height) / 2

        # Clamp position to keep within image bounds
        new_x = max(0, min(new_x, img_width - new_width))
        new_y = max(0, min(new_y, img_height - new_height))

        r["x"] = new_x
        r["y"] = new_y
        r["width"] = new_width
        r["height"] = new_height

        self.rect = r

    def render(self):
        """Render crop overlay using ImGui draw list"""
        if not self.active or not self.rect:
            return

        # Aspect ratio dropdown window
        imgui.set_next_window_position(10, 10)
        imgui.begin(
            "Crop Aspect Ratio",
            flags=imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_MOVE,
        )

        current_name = self.aspect_ratios[self.aspect_ratio_idx][0]
        if imgui.begin_combo("Aspect Ratio", current_name):
            for i, (name, ratio) in enumerate(self.aspect_ratios):
                is_selected = i == self.aspect_ratio_idx
                if imgui.selectable(name, is_selected)[0]:
                    old_idx = self.aspect_ratio_idx
                    self.aspect_ratio_idx = i
                    # Apply aspect ratio constraint to current crop rect
                    if old_idx != i and ratio is not None:
                        self.apply_aspect_ratio(ratio)
                        self.validate_rect()
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.end()

        draw_list = imgui.get_foreground_draw_list()
        r = self.rect

        # Convert crop rect corners to screen coords
        tl = self.image_to_screen_coords((r["x"], r["y"]))
        br = self.image_to_screen_coords((r["x"] + r["width"], r["y"] + r["height"]))

        if not tl or not br:
            return

        # Draw crop border
        border_color = imgui.get_color_u32_rgba(1, 1, 1, 1)
        draw_list.add_rect(tl[0], tl[1], br[0], br[1], border_color, thickness=2.0)

        # Draw handles
        handle_size = 8
        handle_color = imgui.get_color_u32_rgba(1, 1, 1, 1)

        handles = {
            "tl": (r["x"], r["y"]),
            "tr": (r["x"] + r["width"], r["y"]),
            "bl": (r["x"], r["y"] + r["height"]),
            "br": (r["x"] + r["width"], r["y"] + r["height"]),
            "t": (r["x"] + r["width"] / 2, r["y"]),
            "b": (r["x"] + r["width"] / 2, r["y"] + r["height"]),
            "l": (r["x"], r["y"] + r["height"] / 2),
            "r": (r["x"] + r["width"], r["y"] + r["height"] / 2),
        }

        for img_pos in handles.values():
            screen_pos = self.image_to_screen_coords(img_pos)
            if screen_pos:
                draw_list.add_circle_filled(
                    screen_pos[0], screen_pos[1], handle_size, handle_color
                )

        # Draw grid (rule of thirds)
        grid_color = imgui.get_color_u32_rgba(1, 1, 1, 0.3)
        for i in [1, 2]:
            # Vertical lines
            x = r["x"] + r["width"] * i / 3
            p1 = self.image_to_screen_coords((x, r["y"]))
            p2 = self.image_to_screen_coords((x, r["y"] + r["height"]))
            if p1 and p2:
                draw_list.add_line(p1[0], p1[1], p2[0], p2[1], grid_color, 1.0)

            # Horizontal lines
            y = r["y"] + r["height"] * i / 3
            p1 = self.image_to_screen_coords((r["x"], y))
            p2 = self.image_to_screen_coords((r["x"] + r["width"], y))
            if p1 and p2:
                draw_list.add_line(p1[0], p1[1], p2[0], p2[1], grid_color, 1.0)
