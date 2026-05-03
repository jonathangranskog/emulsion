import glfw
import glm
import torch
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs
import imgui.integrations.glfw
import OpenGL.GL as gl
from typing import Callable, Any
from contextlib import contextmanager

from src.core.metadata import ImageMetadata
from src.interface.texture import TextureManager
from src.interface.quad_renderer import QuadRenderer
from src.interface.crop_tool import CropTool
from src.search.window import ChatWindow
from src.effects.base import ImageEffect
from src.pipeline.stack import EffectStack
from src.io.image_saver import ImageSaver
from src.pipeline.torch_processor import TorchEffectsProcessor
from src.utils.conversion import expand_to_aspect_ratio, resize_image

ZOOM_FACTOR = 1.05
ZOOM_MIN = 0.1
ZOOM_MAX = 10.0
TRANSLATION_SCALE = 200.0


class ImageViewer:
    def __init__(
        self,
        window: glfw._GLFWwindow,
        on_frame_callback: Callable[[], None] = None,
        effects_stack: EffectStack = None,
        torch_effects_processor: TorchEffectsProcessor = None,
        shader_effects_processor=None,
        texture: str = None,
        gl_context=None,
        image_path: str | None = None,
    ):
        self.image_width = 0
        self.image_height = 0
        self.main_texture = texture
        self.gl_context = gl_context
        imgui.create_context()
        self.window = window
        self.impl = imgui.integrations.glfw.GlfwRenderer(window)

        # Effects stack and all processors
        self.effects_stack = effects_stack
        self.effects_processors = {
            "torch": torch_effects_processor,
            "shader": shader_effects_processor,
        }
        self.processor_mode = "shader"  # Default to shader
        self.quad_renderer = QuadRenderer(
            self.effects_processor.fragment_shader_string()
        )
        self.background_color = (0.75, 0.75, 0.75, 1.0)

        # Aspect ratio options for save
        self.aspect_ratio_options = [
            ("Original", None),
            ("1:1 (Square)", 1.0),
            ("4:3", 4 / 3),
            ("5:4", 5 / 4),
            ("7:6", 7 / 6),
            ("16:9", 16 / 9),
            ("21:9", 21 / 9),
            ("2:3 (Portrait)", 2 / 3),
            ("4:5 (Portrait)", 4 / 5),
            ("6:7 (Portrait)", 6 / 7),
            ("9:16 (Portrait)", 9 / 16),
        ]
        self.selected_aspect_ratio_idx = 0  # Default to original
        self.x_offset = 0.0  # Horizontal offset ratio (-1 to 1) for saved image
        self.y_offset = 0.0  # Vertical offset ratio (-1 to 1) for saved image
        self.resize_percentage = (
            100  # Resize percentage for saved image (e.g., 50 = 50%, 200 = 200%)
        )

        self.zoom = 1.0
        self.camera_pos = glm.vec2(0.0, 0.0)
        self.is_dragging = False
        self.last_mouse_pos = glm.vec2(0.0, 0.0)

        # Search window
        self.search_window = ChatWindow(effects_stack, image_path=image_path)

        # Parameter drag tracking for undo
        self.param_drag_active = False
        self.param_before_drag = None
        self.param_drag_effect = None
        self.param_drag_name = None

        # Crop tool
        self.crop_tool = CropTool(self)

    def save_image(self):
        # Save from current processor mode (supports both torch and shader)
        image_tensor = self.effects_processor.read_output_as_tensor()

        # Apply resize if not 100%
        if self.resize_percentage != 100:
            image_tensor = resize_image(image_tensor, self.resize_percentage)

        # Apply aspect ratio expansion if selected (with optional offsets)
        _, aspect_ratio = self.aspect_ratio_options[self.selected_aspect_ratio_idx]
        if aspect_ratio is not None:
            # Use the background color for expansion
            bg_color = (
                self.background_color[0],
                self.background_color[1],
                self.background_color[2],
            )
            image_tensor = expand_to_aspect_ratio(
                image_tensor,
                aspect_ratio,
                bg_color,
                self.x_offset,
                self.y_offset,
            )

        ImageSaver.save_image(image_tensor)

    def handle_input(self, io):
        # Handle keyboard shortcuts
        if io.key_ctrl or io.key_super:  # Ctrl or Cmd key
            # Undo: Cmd+Z or Ctrl+Z
            if imgui.is_key_pressed(ord("Z")):
                if io.key_shift:
                    # Redo: Cmd+Shift+Z or Ctrl+Shift+Z
                    if self.effects_stack.redo():
                        self.search_window.add_system_message("Redo successful")
                    else:
                        self.search_window.add_system_message("Nothing to redo")
                else:
                    # Undo: Cmd+Z or Ctrl+Z
                    if self.effects_stack.undo():
                        self.search_window.add_system_message("Undo successful")
                    else:
                        self.search_window.add_system_message("Nothing to undo")
                return

            # Save the image: Cmd+S or Ctrl+S
            if imgui.is_key_pressed(ord("S")):
                self.save_image()
                return

        # Toggle effects bypass: F key (only if not typing in text input)
        if (
            imgui.is_key_pressed(ord("F"))
            and not self.crop_tool.active
            and not io.want_capture_keyboard
        ):
            self._toggle_effects_bypass()
            return

        # Toggle crop mode: C key (only if not typing in text input)
        if (
            imgui.is_key_pressed(ord("C"))
            and not self.crop_tool.active
            and not io.want_capture_keyboard
        ):
            self.crop_tool.activate()
            return

        # Exit crop mode: Escape key
        # Use GLFW to check escape key and track state to detect "just pressed"
        escape_pressed = glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS
        if (
            self.crop_tool.active
            and escape_pressed
            and not getattr(self, "_escape_was_pressed", False)
        ):
            self.crop_tool.cancel()
            self._escape_was_pressed = True
            return
        if not escape_pressed:
            self._escape_was_pressed = False

        # Apply crop: Enter/Return key
        enter_pressed = glfw.get_key(self.window, glfw.KEY_ENTER) == glfw.PRESS
        if (
            self.crop_tool.active
            and enter_pressed
            and not getattr(self, "_enter_was_pressed", False)
        ):
            self.crop_tool.apply()
            self._enter_was_pressed = True
            return
        if not enter_pressed:
            self._enter_was_pressed = False

        # In crop mode, handle crop interactions
        if self.crop_tool.active:
            self.crop_tool.handle_input(io)
            return

        # Ignore camera input when ImGui is using the mouse (windows, sliders, etc.)
        if getattr(io, "want_capture_mouse", False):
            self.is_dragging = False
            return

        mouse_pos = glm.vec2(io.mouse_pos.x, io.mouse_pos.y)

        # Start dragging
        if io.mouse_down[0] and not self.is_dragging:
            self.is_dragging = True
            self.last_mouse_pos = mouse_pos

        # Stop dragging
        if not io.mouse_down[0]:
            self.is_dragging = False

        # Handle dragging
        if self.is_dragging:
            delta = mouse_pos - self.last_mouse_pos
            self.camera_pos.x += delta.x / TRANSLATION_SCALE
            self.camera_pos.y -= delta.y / TRANSLATION_SCALE
            self.last_mouse_pos = mouse_pos

        # Handle zoom
        if io.mouse_wheel != 0:
            zoom_factor = ZOOM_FACTOR if io.mouse_wheel > 0 else 1.0 / ZOOM_FACTOR
            self.zoom = max(ZOOM_MIN, min(self.zoom * zoom_factor, ZOOM_MAX))

    def set_main_texture_parameters(self):
        """Update viewer dimensions from current processor output"""
        # In crop mode, use source dimensions so coordinates align
        if self.crop_tool.active:
            source_tensor = self.effects_processor.source.tensor
            _, self.image_height, self.image_width = source_tensor.shape
        else:
            # Get dimensions directly from processor
            self.image_width, self.image_height = (
                self.effects_processor.get_output_dimensions()
            )

    def create_mvp(self, max_height=720, max_width=1280):
        # Set up projection matrix
        aspect_ratio = max_width / max_height
        projection = glm.ortho(-aspect_ratio, aspect_ratio, -1.0, 1.0, -1.0, 1.0)
        view = glm.translate(
            glm.mat4(1.0), glm.vec3(self.camera_pos.x, self.camera_pos.y, 0.0)
        )
        view = glm.scale(view, glm.vec3(self.zoom, self.zoom, 1.0))

        # Set up model matrix (scale quad to image aspect ratio)
        if self.image_width > 0 and self.image_height > 0:
            image_aspect = self.image_width / self.image_height
            if image_aspect > 1.0:
                model = glm.scale(glm.mat4(1.0), glm.vec3(image_aspect, 1.0, 1.0))
            else:
                model = glm.scale(glm.mat4(1.0), glm.vec3(1.0, 1.0 / image_aspect, 1.0))
        else:
            model = glm.mat4(1.0)

        return projection, view, model

    @property
    def effects_processor(self):
        return self.effects_processors[self.processor_mode]

    def render(self, max_height=720, max_width=1280):
        # Use actual framebuffer size for proper window resize handling
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)

        # Render the quad with the texture
        projection, view, model = self.create_mvp(fb_height, fb_width)

        # In crop mode, show source image (without effects) so coordinates align
        if self.crop_tool.active:
            texture_id = TextureManager.get_texture_id(self.main_texture)
        else:
            # Get the correct texture to render - for FBO mode, this returns the FBO output directly
            # whereas for other effects processors it returns the main texture
            texture_id = self.effects_processor.get_output_texture_id()

        self.quad_renderer.render(
            texture_id,
            projection,
            view,
            model,
            self.effects_processor.upload_uniforms,
        )

        # Render the effect controls
        if self.effects_stack is not None:
            imgui.begin("Effects")

            self.render_processor_selection()

            imgui.separator()

            # Effects bypass toggle
            if self.effects_stack.effects_bypassed:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.5, 0.0, 1.0)
                if imgui.button("Effects OFF (F)"):
                    self._toggle_effects_bypass()
                imgui.pop_style_color()
            else:
                if imgui.button("Effects ON (F)"):
                    self._toggle_effects_bypass()

            imgui.separator()

            # Crop tool button
            if not self.crop_tool.active:
                if imgui.button("Crop Image (C)"):
                    self.crop_tool.activate()
            else:
                imgui.text("Crop Mode Active")
                imgui.same_line()
                if imgui.button("Apply (Enter)"):
                    self.crop_tool.apply()
                imgui.same_line()
                if imgui.button("Cancel (Esc)"):
                    self.crop_tool.cancel()

            imgui.separator()

            for idx, eff in enumerate(self.effects_stack.effects):
                self.create_effect_ui(eff, idx)

            # Background color picker
            imgui.separator()
            imgui.text("Background Color")
            changed, new_color = imgui.color_edit3(
                "##bg_color",
                self.background_color[0],
                self.background_color[1],
                self.background_color[2],
            )
            if changed:
                self.background_color = (*new_color, 1.0)  # Keep alpha at 1.0

            # Aspect ratio selection for save
            imgui.separator()
            imgui.text("Save Aspect Ratio")
            aspect_ratio_names = [name for name, _ in self.aspect_ratio_options]
            current_name = aspect_ratio_names[self.selected_aspect_ratio_idx]
            if imgui.begin_combo("##aspect_ratio", current_name):
                for idx, (name, _) in enumerate(self.aspect_ratio_options):
                    is_selected = idx == self.selected_aspect_ratio_idx
                    if imgui.selectable(name, is_selected)[0]:
                        self.selected_aspect_ratio_idx = idx
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            # Image offset sliders
            imgui.text("Image Offset")
            changed_x, self.x_offset = imgui.slider_float(
                "X Offset##x_offset", self.x_offset, -1.0, 1.0
            )
            imgui.same_line()
            if imgui.button("Reset##x_offset_reset"):
                self.x_offset = 0.0

            changed_y, self.y_offset = imgui.slider_float(
                "Y Offset##y_offset", self.y_offset, -1.0, 1.0
            )
            imgui.same_line()
            if imgui.button("Reset##y_offset_reset"):
                self.y_offset = 0.0

            # Resize percentage slider
            imgui.text("Resize Percentage")
            changed_resize, self.resize_percentage = imgui.slider_int(
                "Resize %##resize_percentage", self.resize_percentage, 10, 300
            )
            imgui.same_line()
            if imgui.button("Reset##resize_percentage_reset"):
                self.resize_percentage = 100

            # Add save button
            if imgui.button("Save Image (Cmd+S)"):
                self.save_image()
            imgui.end()

        # Render the imgui interface
        imgui.begin("Debug Info")
        if imgui.tree_node("Image Information"):
            imgui.text(f"Dimensions: {self.image_width} x {self.image_height}")
            imgui.text(
                f"Texture ID: {TextureManager.get_texture_id(self.main_texture)}"
            )
            imgui.tree_pop()

        # Display EDR/HDR status
        if self.gl_context and imgui.tree_node("Display Information"):
            if self.gl_context.is_edr_enabled():
                imgui.text_colored("EDR Status: Enabled", 0.2, 1.0, 0.2, 1.0)  # Green
                headroom = self.gl_context.get_edr_headroom()
                imgui.text(f"Max Headroom: {headroom:.1f}x SDR white")
                max_nits = headroom * 100  # Assuming 100 nits = SDR white
                imgui.text(f"Peak Brightness: ~{max_nits:.0f} nits")
                imgui.text("Values >1.0 will display brighter")
            else:
                imgui.text_colored(
                    "EDR Status: Disabled (SDR)", 1.0, 0.5, 0.2, 1.0
                )  # Orange
                imgui.text("Values >1.0 will be clamped")
            imgui.tree_pop()

        # Display Camera Metadata
        metadata = ImageMetadata.all()
        if metadata and imgui.tree_node("Camera Metadata"):
            # Shutter Speed (Exposure Time)
            if "EXIF ExposureTime" in metadata:
                exposure = metadata["EXIF ExposureTime"]
                if exposure.values:  # Check values list is non-empty
                    # Format as fraction if less than 1 second
                    if hasattr(exposure.values[0], "num"):
                        num = exposure.values[0].num
                        denom = exposure.values[0].den
                        if num > 0 and denom > 0:  # Validate non-zero
                            if num < denom:
                                # Simplify fraction to 1/X format for readability
                                simplified_denom = round(denom / num)
                                imgui.text(f"Shutter Speed: 1/{simplified_denom}s")
                            else:
                                imgui.text(f"Shutter Speed: {num / denom:.2f}s")
                    else:
                        imgui.text(f"Shutter Speed: {exposure.values[0]}s")

            # ISO
            iso_tags = ["EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity"]
            for tag in iso_tags:
                if tag in metadata:
                    iso = metadata[tag]
                    if iso.values:  # Check values list is non-empty
                        imgui.text(f"ISO: {iso.values[0]}")
                    break

            # Aperture (F-Number)
            if "EXIF FNumber" in metadata:
                fnumber = metadata["EXIF FNumber"]
                if fnumber.values:  # Check values list is non-empty
                    if hasattr(fnumber.values[0], "num"):
                        num = fnumber.values[0].num
                        den = fnumber.values[0].den
                        if den > 0:  # Validate non-zero denominator
                            f_value = num / den
                            imgui.text(f"Aperture: f/{f_value:.1f}")
                    else:
                        imgui.text(f"Aperture: f/{fnumber.values[0]}")

            # As-shot color temperature (from RAW white balance metadata)
            if "raw_as_shot_kelvin" in metadata:
                kelvin = metadata["raw_as_shot_kelvin"]
                imgui.text(f"Color Temp: {kelvin:.0f} K")

            imgui.tree_pop()

        imgui.end()

        # Render the search window
        self.search_window.render(self.effects_processor)

        # Render crop overlay (after all ImGui windows to draw on top)
        self.crop_tool.render()

    def render_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            imgui.new_frame()
            io = imgui.get_io()
            self.handle_input(io)

            # Set background color to white
            gl.glClearColor(*self.background_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            # Update the selected effects processor
            self.effects_processor.on_frame(self.main_texture)

            # Update viewer dimensions based on processor output (important for shader mode with padding)
            self.set_main_texture_parameters()
            self.render()

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

    def shutdown(self):
        self.impl.shutdown()

    def _toggle_effects_bypass(self):
        """Toggle effects bypass — swap displayed texture without recomputing."""
        bypassed = self.effects_stack.toggle_effects_bypass()
        if bypassed:
            self.effects_processor.prepare_bypass_output(self.main_texture)
        else:
            self.effects_processor.restore_from_bypass(self.main_texture)
        state = "off" if bypassed else "on"
        self.search_window.add_system_message(f"Effects {state}")

    def render_processor_selection(self):
        # Processor mode selection
        imgui.text("Rendering Mode:")
        modes = ["torch", "shader"]
        current_idx = modes.index(self.processor_mode)

        changed = False
        for i, mode in enumerate(modes):
            selected = imgui.radio_button(mode, current_idx == i)
            if selected and i != current_idx:
                self.processor_mode = mode
                changed = True
            if i < len(modes) - 1:
                imgui.same_line()

        if changed:
            self.quad_renderer.build_shaders(
                self.effects_processor.fragment_shader_string()
            )
            self.effects_stack.mark_reconstruction_required()

            # Always reset texture to original image when switching modes
            # This ensures we start fresh and don't have accumulated effects
            source_tensor = self.effects_processor.source.tensor
            TextureManager.upload_texture_tensor(source_tensor, self.main_texture)
            self.set_main_texture_parameters()
        return changed

    def render_slider_parameter(
        self, p: dict, eff: ImageEffect, idx: int, name: str, cur: Any
    ):
        """Render float, int, or vec3 slider parameters with unified logic."""
        effect_type = p["type"]
        default_val = p["default"]

        # Handle vec3 special case (tensor conversion)
        if effect_type == "vec3":
            if isinstance(cur, list):
                cur = torch.tensor(cur)
            cur_cpu = cur.cpu()
            is_modified = self.is_modified(cur, default_val)

            # Render color picker
            with self.modified_color_context(is_modified):
                changed, new_color = imgui.color_edit3(
                    f"##{idx}_{name}",
                    float(cur_cpu[0]),
                    float(cur_cpu[1]),
                    float(cur_cpu[2]),
                )

            # Track drag and handle changes
            self.track_drag_state_activated(eff, name, cur.clone())
            if changed:
                new_value = torch.tensor(new_color, dtype=cur.dtype, device=cur.device)
                self.slider_changed(p, eff, name, new_value)
            self.capture_drag_state_ended(eff, name)

        else:
            # Handle float and int sliders
            pmin = p["min"]
            pmax = p["max"]
            is_modified = self.is_modified(cur, default_val)

            # Select appropriate cast and slider functions
            cast_func = float if effect_type == "float" else int
            slider_func = (
                imgui.slider_float if effect_type == "float" else imgui.slider_int
            )

            # Render slider
            with self.modified_color_context(is_modified):
                changed, new_val = slider_func(
                    f"##{idx}_{name}",
                    cast_func(cur),
                    cast_func(pmin),
                    cast_func(pmax),
                )

            # Track drag and handle changes
            self.track_drag_state_activated(eff, name, cast_func(cur))
            if changed:
                self.slider_changed(p, eff, name, cast_func(new_val))
            self.capture_drag_state_ended(eff, name)

    def render_button_parameter(
        self, p: dict, eff: ImageEffect, label: str, idx: int, name: str
    ):
        """Render button parameters."""
        if imgui.button(f"{label}##{idx}_{name}"):
            callback = p["callback"]
            if callable(callback):
                # TODO: Pass current image instead of source?
                original_image = self.effects_processors["torch"].source.tensor
                callback(original_image)
            self.reconstruction_required(f"{type(eff).__name__}.{label} action")

    def render_text_parameter(
        self, p: dict, eff: ImageEffect, label: str, idx: int, name: str, cur: str
    ):
        """Render text input parameters."""
        enter_pressed, new_text = imgui.input_text(
            f"##{label}_{idx}_{name}",
            cur,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        if enter_pressed:
            setattr(eff, name, new_text)
            self.reconstruction_required(
                f"Change {type(eff).__name__}.{name} to {new_text}"
            )

    def render_choice_parameter(
        self, p: dict, eff: ImageEffect, label: str, idx: int, name: str, cur: int
    ):
        """Render dropdown/combo box choice parameters."""
        choices = p.get("choices", [])
        if not choices:
            imgui.text("No choices available")
            return

        # Get current selection name
        current_name = choices[cur] if 0 <= cur < len(choices) else "Invalid"

        # Render combo box
        if imgui.begin_combo(f"##{idx}_{name}", current_name):
            for i, choice in enumerate(choices):
                is_selected = i == cur
                if imgui.selectable(choice, is_selected)[0]:
                    setattr(eff, name, i)
                    self.reconstruction_required(
                        f"Change {type(eff).__name__}.{name} to {choice}"
                    )
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

    def render_bool_parameter(
        self, p: dict, eff: ImageEffect, label: str, idx: int, name: str, cur: bool
    ):
        """Render checkbox for boolean parameters."""
        clicked, new_value = imgui.checkbox(f"##{idx}_{name}", cur)
        if clicked:
            setattr(eff, name, new_value)
            self.reconstruction_required(
                f"Change {type(eff).__name__}.{name} to {new_value}"
            )

    def render_file_parameter(
        self, p: dict, eff: ImageEffect, label: str, idx: int, name: str, cur: str
    ):
        """Render file picker parameters."""
        # Display current file path (read-only)
        imgui.input_text(f"##{idx}_{name}_display", cur, imgui.INPUT_TEXT_READ_ONLY)

        # File picker button
        imgui.same_line()
        if imgui.button(f"Browse##{idx}_{name}"):
            result = self.render_file_dialog(p, label)
            if result and len(result) > 0:
                filename = result[0]
                if hasattr(eff, "on_file_load"):
                    eff.on_file_load(filename)
                else:
                    setattr(eff, name, filename)
                self.reconstruction_required(
                    f"Load file for {type(eff).__name__}.{name}"
                )

    def create_effect_ui(self, eff: ImageEffect, idx: int):
        title = eff.__class__.__name__
        n_effects = len(self.effects_stack.effects)

        opened = imgui.tree_node(f"{title}##eff_{id(eff)}")

        # Small up/down reorder buttons on the same line as the tree node header
        imgui.same_line()
        imgui.set_window_font_scale(0.7)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 0))
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3 if idx == 0 else 1.0)
        if imgui.arrow_button(f"##up_{idx}", imgui.DIRECTION_UP) and idx > 0:
            if self.effects_stack.move_effect(idx, -1):
                self.effects_stack.capture_state(f"Move {type(eff).__name__} up")
        imgui.pop_style_var()
        imgui.same_line()
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3 if idx == n_effects - 1 else 1.0)
        if (
            imgui.arrow_button(f"##down_{idx}", imgui.DIRECTION_DOWN)
            and idx < n_effects - 1
        ):
            if self.effects_stack.move_effect(idx, 1):
                self.effects_stack.capture_state(f"Move {type(eff).__name__} down")
        imgui.pop_style_var()
        imgui.pop_style_var()
        imgui.set_window_font_scale(1.0)

        if opened:
            toggled = eff.toggled
            imgui.push_style_var(
                imgui.STYLE_FRAME_PADDING, (0, 0)
            )  # Less padding to make it smaller
            toggle_changed, toggle_value = imgui.checkbox(f"##{idx}_{toggled}", toggled)
            imgui.pop_style_var()
            imgui.same_line()
            imgui.text("Toggle Effect")

            if toggle_changed:
                eff.toggled = toggle_value
                self.effects_stack.mark_values_changed()
                self.effects_stack.capture_state(f"Toggle {type(eff).__name__}")

            # Loop over parameters
            for p in eff.get_params():
                name = p["name"]
                label = p["label"]
                effect_type = p["type"]
                cur = getattr(eff, name)
                imgui.text(label)

                if effect_type in ("float", "int", "vec3"):
                    self.render_slider_parameter(p, eff, idx, name, cur)
                elif effect_type == "button":
                    self.render_button_parameter(p, eff, label, idx, name)
                elif effect_type == "file":
                    self.render_file_parameter(p, eff, label, idx, name, cur)
                elif effect_type == "text":
                    self.render_text_parameter(p, eff, label, idx, name, cur)
                elif effect_type == "choice":
                    self.render_choice_parameter(p, eff, label, idx, name, cur)
                elif effect_type == "bool":
                    self.render_bool_parameter(p, eff, label, idx, name, cur)

                # Add reset button next to slider
                imgui.same_line()
                if imgui.button(f"Reset##{idx}_{name}"):
                    default_val = p["default"]
                    setattr(eff, name, default_val)
                    # This will cause the effects to be reconstructed
                    eff.reset_file()
                    self.reconstruction_required(f"Reset {type(eff).__name__}.{name}")

            imgui.tree_pop()

    def track_drag_state_activated(self, eff: ImageEffect, name: str, value: Any):
        if imgui.is_item_activated():
            self.param_drag_active = True
            self.param_before_drag = value
            self.param_drag_effect = eff
            self.param_drag_name = name

    def is_modified(self, current: Any, other: Any) -> bool:
        if isinstance(current, torch.Tensor):
            if not isinstance(other, torch.Tensor):
                other = torch.tensor(other, dtype=current.dtype, device=current.device)
            return (torch.abs(current - other) > 0.001).any()
        elif isinstance(current, list):
            return current != other
        elif isinstance(current, float):
            return abs(current - other) > 0.001
        elif isinstance(current, int):
            return abs(current - other) > 0
        else:
            return current != other

    def capture_drag_state_ended(self, eff: ImageEffect, name: str):
        # Capture state on drag end if value changed
        if imgui.is_item_deactivated() and self.param_drag_active:
            new_value = getattr(eff, name)
            if self.is_modified(new_value, self.param_before_drag):
                self.effects_stack.capture_state(f"Change {type(eff).__name__}.{name}")
            self.param_drag_active = False

    @contextmanager
    def modified_color_context(self, is_modified: bool):
        """Context manager for applying modified color styling to ImGui elements."""
        if is_modified:
            imgui.push_style_color(
                imgui.COLOR_FRAME_BACKGROUND, 0.3, 0.3, 0.7, 1.0
            )  # Blue tint
            imgui.push_style_color(
                imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.4, 0.4, 0.8, 1.0
            )  # Brighter blue on hover
            imgui.push_style_color(
                imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.5, 0.5, 0.9, 1.0
            )  # Even brighter when active
        try:
            yield
        finally:
            # Pop all 3 colors we pushed
            if is_modified:
                imgui.pop_style_color(3)

    def render_file_dialog(self, params: dict[str, Any], label: str):
        # Use portable file dialogs for simpler integration
        # Convert file types to filter strings format
        file_types = params.get("file_types", [("All Files", "*.*")])
        filter_strings = []
        for desc, pattern in file_types:
            filter_strings.append(f"{desc}|{pattern}")

        # Create and show the file dialog
        dialog = portable_file_dialogs.open_file(
            title=f"Select {label}",
            default_path=".",
            filters=filter_strings,
        )

        # Get the result (this blocks until user selects)
        result = dialog.result()
        return result

    def reconstruction_required(self, reason: str = ""):
        self.effects_stack.mark_reconstruction_required()
        self.effects_stack.mark_values_changed()
        self.effects_stack.capture_state(reason)

    def slider_changed(self, params, eff: ImageEffect, name: str, new_value: Any):
        setattr(eff, name, new_value)
        self.effects_stack.mark_values_changed()
        if params.get("requires_reconstruction", False):
            # Drag state captures state for history
            self.effects_stack.mark_reconstruction_required()
