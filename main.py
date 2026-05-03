"""
The launch file for starting up the image editor.

For now, it will just load the sample image file and display it in an imgui window.
"""

import argparse

import torch

from src.pipeline.stack import EffectStack
from src.pipeline.cache import EffectsCache
from src.pipeline.shader_processor import ShaderEffectsProcessor
from src.pipeline.torch_processor import TorchEffectsProcessor
from src.interface.gl_context import GLContext
from src.interface.texture import TextureManager
from src.interface.viewer import ImageViewer
from src.io.image_loader import ImageLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/sample_jpg.jpeg")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument(
        "--seed-cache",
        type=str,
        default=None,
        help="Path to another image whose effects cache will be used as the starting point",
    )
    args = parser.parse_args()

    image = ImageLoader.load_image_safe(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")
    print(f"Loaded the image {args.image} with shape {image.shape}.")

    # Create the effects stack
    stack = EffectStack()
    effects_cache = EffectsCache(args.image, stack) if not args.disable_cache else None
    loaded = False
    if effects_cache is not None:
        loaded = effects_cache.load_from_cache()
        if not loaded and args.seed_cache:
            loaded = effects_cache.load_from_seed(args.seed_cache)
    if not loaded:
        stack.set_effects([])

    # Create the GL context and the window
    gl_context = GLContext()
    window = gl_context.create_window(1280, 720, "Emulsion")

    # Create the texture for the image
    TextureManager.register_texture2d(image.tensor, "main_texture")

    # Create the torch effects processor and its callback
    torch_effecs_processor = TorchEffectsProcessor(image, stack)

    # Create the shader effects processors and their callbacks
    shader_effects_processor = ShaderEffectsProcessor(image, stack)

    # Initialize the viewer and start the render loop
    viewer = ImageViewer(
        window,
        torch_effects_processor=torch_effecs_processor,
        shader_effects_processor=shader_effects_processor,
        effects_stack=stack,
        texture="main_texture",
        gl_context=gl_context,
        image_path=args.image,
    )
    viewer.set_main_texture_parameters()
    stack.mark_reconstruction_required()
    viewer.render_loop()

    # Clean up
    viewer.shutdown()
    gl_context.destroy_window(window)


if __name__ == "__main__":
    main()
