# Emulsion

Emulsion is an AI-powered desktop photo editor built in Python. It combines a real-time GPU-accelerated effect stack with a Gemini-powered agentic interface that collaborates with you to plan, preview, and apply photographic edits.

> **Note:** This repository is provided as reference and for learning. It is not a fully functional product. Many parts of the codebase have been vibecoded so bugs are to be expected. Feel free to flag these if you encounter any. 

## Features

- **Human-in-the-loop agentic editing** — describe a look in natural language; the AI plans a multi-step edit and shows you 5 preview options per step to choose from, with live feedback support
- **22 stackable image effects** — from basic tonal adjustments to film grain, bloom, vignette, and LUTs
- **Text-to-Shader** — generate custom GLSL image effects from a text description
- **Semantic LUT search** — find LUT presets by describing the look you want
- **Dual processing modes** — GPU (GLSL shaders via OpenGL) and CPU (PyTorch tensors)
- **RAW image support** — DNG, CR2, etc. Only tested Leica Q2 DNGs and Canon 60D CR2's. 

> **Note:** Only tested on macOS. Other platforms may work but are unsupported.

## Setup

**Requirements:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/), a [Gemini API key](https://aistudio.google.com/apikey)

```bash
# Install dependencies
uv sync

# Provide your Gemini API key (choose one)
export GEMINI_API_KEY=your_key_here
# or place it in a file:
echo "your_key_here" > .secrets/GEMINI_API_KEY
```

To open an image:

```bash
uv run python main.py --image path/to/photo.dng
```

A Gemini API key is **required** — the editor will not start without one. It is used for agentic editing, text-to-shader, semantic LUT search, and action classification.

## Agentic Interface

The chat panel at the bottom of the editor is the primary way to interact with the AI. Type a description of the edit you want:

```
moody cinematic look with warm shadows
```

The agent will:

1. **Plan** a sequence of effects to achieve the look (up to 6 steps)
2. **Generate 5 candidate presets** per step and render preview thumbnails
3. **Present the previews** for you to pick the one you like best
4. **Continue** to the next step with your chosen parameters

At any step you can:
- **Click a preview** to select it and advance
- **Type feedback** to ask the agent to regenerate candidates with adjustments (e.g. "less warm", "more contrast")
- **Skip** the step entirely
- **Stop** the agent and keep the edits applied so far

### Reference Image

Click **Ref** to upload a reference image. The agent will use it as a visual target — useful for matching the color grading or mood of a specific photo.

### Models

The **Model** selector in the chat panel controls which Gemini model handles planning:
- **Auto** — uses Flash for 1–2 step plans, Pro for longer ones
- **Flash** — faster and cheaper, good for simple edits (default)
- **Pro** — higher quality reasoning for complex multi-step edits

### Fast Commands

Certain inputs bypass the AI entirely for immediate actions:

| Input | Action |
|---|---|
| `undo` / `redo` | Undo or redo last change |
| `clear` | Remove all effects |
| `remove [effect]` | Remove the last instance of a named effect |
| `t2s: <description>` | Generate a custom GLSL shader from text |
| `search: <query>` or `lut: <query>` | Semantic LUT search |
| `[effect name]` | Add a named effect directly (e.g. `grain`, `vignette`) |

## Semantic LUT Search

Semantic LUT Search lets you find and apply a LUT preset by describing the look you want in plain English:

```
search: faded vintage film with warm highlights
```

or equivalently:

```
lut: dark and moody cinematic look
```

When a query is received, it is embedded with Gemini's `gemini-embedding-001` model and compared against a precomputed cache of LUT embeddings using cosine similarity. The closest matching LUT is applied automatically as a Search LUT effect.

The cache stores a short-description embedding for each LUT alongside its tensor data, domain range, and name — so no LUT files need to be on disk at runtime. A minimal fallback cache (`assets/tiny_lut_cache.npy`) is included; see [LUT Library Setup](#lut-library-setup) for building a full cache from your own `.cube` files or downloading a larger library.

## Text-to-Shader

Text-to-Shader (T2S) lets you describe an image effect in plain English and generates a custom GLSL shader for it:

```
t2s: painterly oil painting effect with visible brushstrokes
```

Generated shaders are validated automatically and cached to disk. Failed generation attempts are fed back to the model for up to 3 self-correction rounds.

## Effects

All effects can be stacked in any order. Each has independent parameters controlled via sliders, color pickers, and file selectors in the effect stack panel.

**Tonal & Color:**
| Effect | Description |
|---|---|
| Exposure | Photographic exposure in EV stops (±5) |
| Gamma | Non-linear brightness correction |
| Contrast | Contrast with adjustable midpoint pivot |
| Saturation | ITU-R BT.709 saturation control |
| Vibrance | Vibrance (protects already-saturated colors) |
| Highlights | Luminance-masked adjustment of bright areas |
| Shadows | Luminance-masked adjustment of dark areas |
| Temperature & Tint | Color temperature and tint (white balance) |
| Color Shift | Per-channel color offset with scale control |
| Black & White | Monochrome conversion with filter color |

**Tone Mapping:**
| Effect | Description |
|---|---|
| Tone Mapping | Reinhard tone mapping |
| ACES | ACES Filmic tone mapping |

**Artistic & Creative:**
| Effect | Description |
|---|---|
| Bloom / Glow | Threshold-based bloom with Gaussian blur |
| Vignette | Corner darkening with feather and strength control |
| Grain | Film grain via Gaussian Splatting — Metal GPU accelerated on macOS |
| Gaussian Blur | Full-image blur (radius 1–300) |
| Noise Blur | Turbulent variable blur driven by simplex noise |
| Texture Overlay | Image texture blend with adjustable opacity and blend mode |
| Padding | Colored border with configurable size |
| Crop | Non-destructive image crop |

**LUT & Lookup:**
| Effect | Description |
|---|---|
| LUT | Load an Adobe Cube (.cube) LUT file with strength control |
| Search LUT | Find and apply a LUT by semantic description |
| Text-to-Shader | Custom GLSL effect generated from a text prompt |

## Processing Modes

Switch between modes at any time via the radio buttons in the UI:

- **Shader Mode (GPU)** — runs each effect as its own GLSL fragment shader, ping-ponging between OpenGL FBOs. Best performance for interactive use.
- **Torch Mode (CPU)** — applies each effect sequentially as PyTorch tensor operations. Full float32 precision; useful for debugging or on systems without GPU support.

## Image Loading & Export

**Supported input formats:**
- RAW: DNG, CR2, CR3, NEF, ARW, ORF, RW2, RAF (via rawpy)
- Standard: JPEG, PNG, TIFF, BMP (via Pillow)

RAW loading includes camera white balance extraction, sensor crop margin handling, digital zoom recovery, and all 8 EXIF orientation variants.

> **Note:** RAW support has only been tested with Leica Q2 DNG files. Other RAW formats may work via rawpy but are untested.

**Export:**
- Save with **Ctrl+S** or the Save button — outputs PNG to the `output/` directory with a timestamp
- Resize percentage (10–300%), aspect ratio presets, X/Y offset, and background fill color are available in the export panel

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl/Cmd+S | Save image |
| Ctrl/Cmd+Z | Undo |
| Ctrl/Cmd+Shift+Z | Redo |
| F | Toggle effects bypass (preview original) |
| Mouse drag | Pan |
| Mouse wheel | Zoom (0.1×–10×) |

## Session Persistence

The effect stack is automatically saved to disk (keyed by image path) so your edits persist between sessions. Chat history is also saved per image.

To start a session with another image's effects as a baseline:

```bash
uv run python main.py --image new_photo.dng --seed-cache reference_photo.dng
```

To disable caching entirely:

```bash
uv run python main.py --disable-cache
```

## LUT Library Setup

The semantic LUT search requires a precomputed embedding cache. If you have a library of `.cube` LUT files, you can build the cache with the scripts in `scripts/`:

```bash
# Caption LUTs in `assets/luts` with Gemini vision
uv run python scripts/caption_all_luts.py

# Generate embeddings from captions in `assets/captions`
uv run python scripts/create_lut_embeddings.py

# Build the search cache from embeddings in `assets/lut_embeddings`
uv run python scripts/construct_lut_search_cache.py
```

A minimal fallback cache (`assets/tiny_lut_cache.npy`) is included for testing without a full LUT library.

Alternatively, you can download a prebuilt larger LUT cache directly:

```bash
# Install gdown if needed
pip install gdown

# Download to the assets directory
gdown 1rt2ee_uv8fC4dkXe7kpfj92XSfC4l7y8 -O assets/lut_cache.npy
```

The file is also available at this [Google Drive link](https://drive.google.com/file/d/1rt2ee_uv8fC4dkXe7kpfj92XSfC4l7y8/view?usp=drive_link).
