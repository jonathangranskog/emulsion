import platform
from ctypes import c_void_p
import glfw
import OpenGL.GL as gl
import platform


class GLContext:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Set OpenGL version hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        # Request higher bit depth for HDR/EDR support
        # 16-bit per channel allows for values beyond [0,1]
        glfw.window_hint(glfw.RED_BITS, 16)
        glfw.window_hint(glfw.GREEN_BITS, 16)
        glfw.window_hint(glfw.BLUE_BITS, 16)
        glfw.window_hint(glfw.ALPHA_BITS, 16)

        self.is_macos = platform.system() == "Darwin"
        self.edr_enabled = False
        self.max_edr_headroom = 1.0

    def create_window(self, width: int, height: int, title: str) -> glfw._GLFWwindow:
        window = glfw.create_window(width, height, title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)

        # Enable EDR on macOS if available
        if self.is_macos:
            self._enable_macos_edr(window)
            if not self.edr_enabled:
                self._set_macos_window_colorspace_srgb(window)

        return window

    def _enable_macos_edr(self, window: glfw._GLFWwindow):
        """Enable Extended Dynamic Range on macOS."""
        print("=== EDR Detection Debug ===")
        try:
            # Import macOS-specific modules
            print("Attempting to import PyObjC modules...")
            from Cocoa import NSScreen
            import objc

            print("✓ PyObjC modules imported successfully")

            # Query the maximum EDR headroom available on the main screen
            print("Querying NSScreen for EDR capabilities...")
            screen = NSScreen.mainScreen()
            print(f"✓ Got main screen: {screen}")

            self.max_edr_headroom = (
                screen.maximumExtendedDynamicRangeColorComponentValue()
            )
            print(f"✓ Max EDR headroom value: {self.max_edr_headroom}")

            if self.max_edr_headroom > 1.0:
                print(
                    f"✓ HDR display detected! Max EDR headroom: {self.max_edr_headroom:.2f}x SDR white"
                )

                # Get the native NSWindow from GLFW
                print("Getting native NSWindow from GLFW...")
                ns_window_ptr = glfw.get_cocoa_window(window)
                print(f"✓ NSWindow pointer: {ns_window_ptr}")

                ns_window = objc.objc_object(c_void_p=ns_window_ptr)
                print(f"✓ NSWindow object: {ns_window}")

                # Get the content view and its layer
                print("Getting content view...")
                ns_view = ns_window.contentView()
                print(f"✓ NSView: {ns_view}")

                # Enable extended dynamic range content
                print("Setting wantsExtendedDynamicRangeContent to True...")
                ns_view.setWantsExtendedDynamicRangeContent_(True)

                # Verify it was set
                edr_enabled = ns_view.wantsExtendedDynamicRangeContent()
                print(f"✓ wantsExtendedDynamicRangeContent is now: {edr_enabled}")

                self.edr_enabled = True
                print(
                    "✓✓✓ EDR ENABLED - values >1.0 will display as brighter luminance"
                )
            else:
                print(
                    f"✗ SDR display detected - Max headroom is {self.max_edr_headroom:.2f}x (need >1.0)"
                )
                print("  EDR not available (values will be clamped to [0,1])")

        except ImportError as e:
            print(f"✗ ImportError: {e}")
            print("  PyObjC not installed. EDR support disabled.")
            print("  Install with: pip install pyobjc-framework-Cocoa")
        except AttributeError as e:
            print(f"✗ AttributeError: {e}")
            print(
                "  This might be a GLFW version issue - get_cocoa_window() may not be available"
            )
        except Exception as e:
            print(f"✗ Unexpected error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

        print("=== EDR Detection Complete ===\n")

    def get_edr_headroom(self) -> float:
        """Get the maximum EDR headroom available (1.0 = SDR, >1.0 = HDR)."""
        return self.max_edr_headroom

    def is_edr_enabled(self) -> bool:
        """Check if EDR is currently enabled."""
        return self.edr_enabled

    def _set_macos_window_colorspace_srgb(self, window: glfw._GLFWwindow):
        """Set the macOS NSWindow color space to sRGB.

        This prevents macOS from interpreting the OpenGL framebuffer as Display P3,
        which would make colors appear more saturated than they actually are.
        See: https://github.com/GLFW/glfw/issues/2748
        """
        try:
            import objc
            from AppKit import NSColorSpace
            from ctypes import c_void_p

            # Get the NSWindow pointer from GLFW
            cocoa_window = glfw.get_cocoa_window(window)
            if not cocoa_window:
                print("Warning: Could not get macOS window handle")
                return

            # Convert the pointer to an ObjC object
            NSWindow = objc.objc_object(c_void_p=cocoa_window)

            # Set the color space to sRGB
            srgb_colorspace = NSColorSpace.sRGBColorSpace()
            NSWindow.setColorSpace_(srgb_colorspace)

            print("Successfully set window color space to sRGB")
        except ImportError as e:
            print(f"Error: {e}")
            print(
                "Warning: pyobjc not available, cannot set window color space to sRGB"
            )
            print(
                "Colors may appear oversaturated. Install pyobjc-framework-Cocoa to fix."
            )
        except Exception as e:
            print(f"Warning: Failed to set window color space: {e}")

    def destroy_window(self, window: glfw._GLFWwindow):
        glfw.destroy_window(window)
        glfw.terminate()
