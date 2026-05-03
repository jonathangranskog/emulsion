"""
This script captions utility LUTs in assets/utility_luts using Gemini.
Unlike regular LUTs, utility LUTs have descriptive filenames that indicate
the operation (e.g., brightness_125, hue_shift_45_degrees).
The system prompt heavily weights the filename for context.
"""

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from src.utils.lut_utils import apply_lut, read_cube_file
from src.core.secrets import get_gemini_api_key
from src.utils.conversion import pil_to_tensor, tensor_to_pil, encode_image_as_jpeg

client = genai.Client(api_key=get_gemini_api_key())


def list_available_luts(lut_dir: str = "assets/utility_luts") -> list[str]:
    """List all available .cube LUT files in the utility_luts folder."""
    lut_path = Path(lut_dir)
    if not lut_path.exists():
        return []
    return sorted([f.name for f in lut_path.glob("*.cube")])


def exponential_delay_gemini_generate_content(
    contents: list,
    config: types.GenerateContentConfig,
    model: str = "gemini-2.5-flash",
    max_retries: int = 10,
    initial_delay: float = 10.0,
    max_delay: float = 1800.0,
):
    """
    Call Gemini's generate_content with exponential backoff for rate limiting and server errors.
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(
                keyword in error_str
                for keyword in [
                    "rate",
                    "quota",
                    "limit",
                    "429",
                    "resource_exhausted",
                    "resourceexhausted",
                ]
            )
            is_server_error = any(
                keyword in error_str
                for keyword in [
                    "500",
                    "502",
                    "503",
                    "504",
                    "internal",
                    "server error",
                    "servererror",
                ]
            )
            is_retryable = is_rate_limit or is_server_error

            if not is_retryable or attempt == max_retries - 1:
                print(f"Error calling Gemini API: {e}")
                raise

            wait_time = min(delay, max_delay)
            error_type = "Rate limit" if is_rate_limit else "Server error"
            print(
                f"{error_type} hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
            delay *= 2


def parse_filename_for_context(filename: str) -> str:
    """
    Parse a utility LUT filename to extract a human-readable description of the operation.

    Examples:
        brightness_125 -> "brightness at 125%"
        hue_shift_-45_degrees -> "hue shift of -45 degrees"
        cool_moderate -> "moderate cool temperature shift"
        desaturate_50 -> "desaturation at 50%"
    """
    name = filename.replace(".cube", "")

    patterns = [
        (r"^brightness_(\d+)$", lambda m: f"brightness adjustment to {m.group(1)}%"),
        (
            r"^exposure_(plus|minus)_(\w+)$",
            lambda m: f"exposure adjustment of {'+' if m.group(1) == 'plus' else '-'}{m.group(2).replace('_', '.')} stops",
        ),
        (
            r"^gamma_(\d+)_(\d+)$",
            lambda m: f"gamma correction of {m.group(1)}.{m.group(2)}",
        ),
        (
            r"^(desaturate|oversaturate)_(\d+)$",
            lambda m: f"{'desaturation' if m.group(1) == 'desaturate' else 'oversaturation'} to {m.group(2)}%",
        ),
        (
            r"^(high|low)_contrast_(\d+)$",
            lambda m: f"{m.group(1)} contrast adjustment to {m.group(2)}%",
        ),
        (
            r"^(warm|cool)_(slight|moderate|strong)$",
            lambda m: f"{m.group(2)} {m.group(1)} color temperature shift",
        ),
        (
            r"^tint_(green|magenta)_(slight|moderate|strong)$",
            lambda m: f"{m.group(2)} {m.group(1)} tint",
        ),
        (
            r"^hue_shift_(-?\d+)_degrees$",
            lambda m: f"hue rotation of {m.group(1)} degrees",
        ),
        (r"^invert$", lambda m: "full color inversion"),
        (r"^invert_(\d+)$", lambda m: f"partial color inversion at {m.group(1)}%"),
        (
            r"^invert_(red|green|blue|luminance)$",
            lambda m: f"{m.group(1)} channel inversion",
        ),
        (
            r"^grayscale_(identity|rec709)$",
            lambda m: f"grayscale conversion using {m.group(1)} method",
        ),
        (
            r"^film_negative_(light|medium|strong)$",
            lambda m: f"{m.group(1)} film negative emulation",
        ),
    ]

    for pattern, formatter in patterns:
        match = re.match(pattern, name)
        if match:
            return formatter(match)

    return name.replace("_", " ")


def get_utility_lut_metadata(lut_path: str) -> dict | None:
    """
    Read the config from a utility LUT's metadata JSON file.

    Returns:
        The config dictionary if found, None otherwise
    """
    metadata_json_path = lut_path.replace(".cube", ".json")

    if not os.path.exists(metadata_json_path):
        return None

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("config")
    except (json.JSONDecodeError, IOError):
        return None


def caption_utility_lut(
    lut_path: str,
    image: Image.Image,
    filename: str,
) -> tuple[str, str]:
    """
    Caption a utility LUT using a single image and filename context.

    Returns:
        Tuple of (technical_description, short_description)
    """
    lut_tensor, domain_min, domain_max = read_cube_file(lut_path)

    image_tensor = pil_to_tensor(image)
    image_with_lut = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)
    image_with_lut = tensor_to_pil(image_with_lut)

    filename_context = parse_filename_for_context(filename)
    config_context = get_utility_lut_metadata(lut_path)

    with open("system_prompts/utility_lut_analysis.txt", "r") as f:
        system_prompt_template = f.read()

    system_prompt = system_prompt_template.format(
        filename=filename,
        config=json.dumps(config_context) if config_context else "Not available",
    )

    response_schema = json.loads(
        open("system_prompts/utility_lut_response_schema.json", "r").read()
    )

    config = types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.3,
        response_mime_type="application/json",
        response_schema=response_schema,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,
        ),
    )

    original_bytes = encode_image_as_jpeg(image)
    lut_applied_bytes = encode_image_as_jpeg(image_with_lut)

    response = exponential_delay_gemini_generate_content(
        contents=[
            f"Utility LUT operation: {filename_context}",
            "Original image:",
            types.Part.from_bytes(
                data=original_bytes,
                mime_type="image/jpeg",
            ),
            "LUT-applied image:",
            types.Part.from_bytes(
                data=lut_applied_bytes,
                mime_type="image/jpeg",
            ),
            system_prompt,
        ],
        config=config,
        model="gemini-2.5-flash",
    )

    try:
        result = json.loads(response.text)
        technical_description = result.get("technical_description", "N/A")
        short_description = result.get("short_description", "N/A")
        return technical_description, short_description, image_with_lut
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response for {filename}. Raw response:")
        print(response.text)
        return "N/A", "N/A", image_with_lut


def update_utility_lut_metadata_json(
    lut_path: str, long_description: str, short_description: str
) -> bool:
    """
    Update the utility LUT's metadata JSON with the generated descriptions.
    """
    metadata_json_path = lut_path.replace(".cube", ".json")

    if not os.path.exists(metadata_json_path):
        return False

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        metadata["descriptions"] = {
            "long": long_description,
            "short": short_description,
        }

        with open(metadata_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Updated metadata JSON: {metadata_json_path}")
        return True

    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Failed to update metadata JSON {metadata_json_path}: {e}")
        return False


def main():
    lut_dir = "assets/utility_luts"
    output_dir = "assets/utility_captions"

    # this is not the image I used for captioning, it's just a sample image
    image = Image.open("assets/sample_jpg.jpeg")

    os.makedirs(output_dir, exist_ok=True)

    lut_files = list_available_luts(lut_dir)
    print(f"Found {len(lut_files)} utility LUTs to process")

    for index, lut_filename in enumerate(lut_files):
        full_lut_path = os.path.join(lut_dir, lut_filename)
        output_path = os.path.join(output_dir, lut_filename.replace(".cube", ".json"))

        if os.path.exists(output_path):
            print(
                f"[{index + 1}/{len(lut_files)}] Skipping {lut_filename} (already captioned)"
            )
            continue

        print(f"[{index + 1}/{len(lut_files)}] Processing {lut_filename}...")

        technical_description, short_description, image_with_lut = caption_utility_lut(
            full_lut_path, image, lut_filename
        )

        update_utility_lut_metadata_json(
            full_lut_path, technical_description, short_description
        )

        config_context = get_utility_lut_metadata(full_lut_path)
        json_data = {
            "short_description": short_description,
            "long_description": technical_description,
            "lut_path": full_lut_path,
            "filename_context": parse_filename_for_context(lut_filename),
            "config": config_context,
        }

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=4)

        image_output_path = output_path.replace(".json", ".jpeg")
        image_with_lut.save(image_output_path)

        print(f"  Saved caption to {output_path}")


if __name__ == "__main__":
    main()
