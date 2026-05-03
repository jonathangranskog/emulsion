"""
This script goes through all the luts in the provided folder and captions them using Gemini for four macbeth images,
saving each JSON caption to a file in the same folder.
"""

import io
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
from google import genai
from google.genai import types
from PIL import Image
import matplotlib.pyplot as plt

from src.utils.lut_utils import apply_lut, read_cube_file
from src.core.secrets import get_gemini_api_key
from src.utils.conversion import pil_to_tensor, tensor_to_pil, encode_image_as_jpeg

client = genai.Client(api_key=get_gemini_api_key())


def list_available_luts(lut_dir: str = "assets/luts") -> list[str]:
    """List all available .cube LUT files."""
    lut_path = Path(lut_dir)
    if not lut_path.exists():
        return []
    return sorted([f.name for f in lut_path.glob("*.cube")])


def expontential_delay_gemini_generate_content(
    contents: list,
    config: types.GenerateContentConfig,
    model: str = "gemini-2.5-flash",
    max_retries: int = 10,
    initial_delay: float = 10.0,
    max_delay: float = 1800.0,
):
    """
    Call Gemini's generate_content with exponential backoff for rate limiting and server errors.

    Args:
        contents: The content to send to Gemini
        config: The generation config
        model: The model name to use
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds

    Returns:
        The response from Gemini

    Raises:
        Exception: If all retries are exhausted or non-retryable error occurs
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
            # Check if it's a retryable error (rate limit or server error)
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
                # If it's not a retryable error or we're out of retries, raise
                print(f"❌ Error calling Gemini API: {e}")
                raise

            # Wait with exponential backoff
            wait_time = min(delay, max_delay)
            error_type = "Rate limit" if is_rate_limit else "Server error"
            print(
                f"⚠️  {error_type} hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
            delay *= 2  # Exponential increase

    raise Exception("Max retries exhausted")


def parse_response(response: str) -> dict:
    # parse a response from Gemini into a dictionary of the information
    try:
        analysis = json.loads(response)
        general_description = analysis.get("general_description", "N/A")
        directors = analysis.get("directors", [])
        movies = analysis.get("movies", [])
        film_emulsions = analysis.get("film_emulsions", [])
        return {
            "general_description": general_description,
            "directors": directors,
            "movies": movies,
            "film_emulsions": film_emulsions,
        }
    except json.JSONDecodeError:
        print("⚠️  Failed to parse JSON response. Raw response:")
        print(response)
        return {
            "general_description": "N/A",
            "directors": [],
            "movies": [],
            "film_emulsions": [],
        }


def caption_lut(lut_path: str, images: list[Image.Image]):
    """
    Caption a LUT for a list of images, returning a list of dictionaries,
    one for each image, containing the caption, movies, directors, and film emulsions.
    """
    lut_tensor, domain_min, domain_max = read_cube_file(lut_path)
    images_with_lut = []

    # Apply LUT to all images
    for image in images:
        image_tensor = pil_to_tensor(image)
        image_tensor = apply_lut(image_tensor, lut_tensor, domain_min, domain_max)
        image_tensor = tensor_to_pil(image_tensor)
        images_with_lut.append(image_tensor)

    # Load the system prompt and schema
    with open("system_prompts/lut_analysis_json.txt", "r") as f:
        system_prompt = f.read()
    response_schema = json.loads(
        open("system_prompts/lut_analysis_response_schema.json", "r").read()
    )

    # Configure generation settings with JSON response
    config = types.GenerateContentConfig(
        max_output_tokens=800,  # Increased for JSON structure
        temperature=0.5,  # Control randomness
        response_mime_type="application/json",  # Force JSON response
        response_schema=response_schema,  # Enforce schema
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,
        ),
    )

    # Send the images to Gemini
    responses = []
    for index, image_with_lut in enumerate(images_with_lut):
        original_bytes = encode_image_as_jpeg(images[index])
        lut_applied_bytes = encode_image_as_jpeg(image_with_lut)
        response = expontential_delay_gemini_generate_content(
            contents=[
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
        responses.append(response.text)

    # Process the responses
    return [parse_response(response) for response in responses], images_with_lut


def get_lut_metadata_prompt(lut_path: str) -> str | None:
    """
    Read the prompt field from a metadata JSON file if it exists.

    Args:
        lut_path: Full path to the .cube LUT file

    Returns:
        The prompt string if found, None otherwise
    """
    metadata_json_path = lut_path.replace(".cube", ".json")

    if not os.path.exists(metadata_json_path):
        return None

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("prompt")
    except (json.JSONDecodeError, IOError):
        return None


def summarize_responses(responses: list[dict], prompt: str | None = None) -> dict:
    """
    Using the responses for each image, summarize the information into a single dictionary
    containing a long prompt and a comma-separated short prompt.

    Args:
        responses: List of response dictionaries from caption_lut
        prompt: Optional prompt string from metadata JSON to consider in summarization
    """
    system_prompt = open("system_prompts/lut_summarization.txt", "r").read()

    contents = []

    # Add prompt information if available
    if prompt:
        contents.append(
            f"Original prompt used to generate this color transformation: {prompt}"
        )
        contents.append("")

    for index, response in enumerate(responses):
        header = f"Color Transformation Description {index + 1}:"
        general_description = response.get("general_description", "N/A")
        directors = response.get("directors", [])
        movies = response.get("movies", [])
        film_emulsions = response.get("film_emulsions", [])
        contents.append(header)
        contents.append(general_description)
        if directors:
            contents.append("Movie Directors:")
            for director in directors:
                contents.append(f"  - {director}")
        if movies:
            contents.append("Movie Titles:")
            for movie in movies:
                contents.append(f"  - {movie}")
        if film_emulsions:
            contents.append("Film Emulsions:")
            for film_emulsion in film_emulsions:
                contents.append(f"  - {film_emulsion}")
        contents.append("")

    contents.append(system_prompt)
    response_schema = json.loads(
        open(
            "system_prompts/lut_summarization_response_schema.json",
            "r",
        ).read(),
    )
    config = types.GenerateContentConfig(
        max_output_tokens=1000,  # Increased for JSON structure
        temperature=0.0,  # Control randomness
        response_mime_type="application/json",  # Force JSON response
        response_schema=response_schema,  # Enforce schema
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,
        ),
    )
    response = expontential_delay_gemini_generate_content(
        contents=contents,
        config=config,
        model="gemini-2.5-flash",
    )
    try:
        json_data = json.loads(response.text)
        long_description = json_data.get("long_description", "N/A")
        short_description = json_data.get("short_description", "N/A")
        return long_description, short_description
    except json.JSONDecodeError:
        print("⚠️  Failed to parse JSON response. Raw response:")
        print(response.text)
        return "N/A", "N/A"


def update_lut_metadata_json(
    lut_path: str, long_description: str, short_description: str
) -> bool:
    """
    Check if a .json file with the same name as the LUT exists in the same directory.
    If it exists, add the descriptions to it and overwrite the file.

    Args:
        lut_path: Full path to the .cube LUT file
        long_description: The long description to add
        short_description: The short description to add

    Returns:
        True if metadata JSON was found and updated, False otherwise
    """
    # Build the path to the potential metadata JSON file
    metadata_json_path = lut_path.replace(".cube", ".json")

    if not os.path.exists(metadata_json_path):
        return False

    try:
        # Read the existing metadata JSON
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        # Add the descriptions as a new key
        metadata["descriptions"] = {
            "long": long_description,
            "short": short_description,
        }

        # Write back the updated metadata
        with open(metadata_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Updated metadata JSON: {metadata_json_path}")
        return True

    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Failed to update metadata JSON {metadata_json_path}: {e}")
        return False


def main():
    lut_dir = "assets/luts"
    images = [
        # this is not the image I used for captioning, it's just a sample image
        Image.open("assets/sample_jpg.jpeg"),
    ]

    for index, lut_path in enumerate(list_available_luts(lut_dir)):
        full_lut_path = os.path.join(lut_dir, lut_path)
        output_path = os.path.join(
            "assets/captions", lut_path.replace(".cube", ".json")
        )
        if os.path.exists(output_path):
            continue
        responses, images_with_lut = caption_lut(full_lut_path, images)

        # Get prompt from metadata JSON if it exists
        metadata_prompt = get_lut_metadata_prompt(full_lut_path)

        long_description, short_description = summarize_responses(
            responses, prompt=metadata_prompt
        )

        # Update the metadata JSON file if it exists alongside the LUT
        update_lut_metadata_json(full_lut_path, long_description, short_description)

        # Save the full caption data to assets/captions
        json_data = {
            "short_description": short_description,
            "long_description": long_description,
            "lut_path": full_lut_path,
            "responses": responses,
        }

        os.makedirs("assets/captions", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=4)

        for idx, image_with_lut in enumerate(images_with_lut):
            image_with_lut.save(f"{output_path.replace('.json', f'_{idx}.jpeg')}")


if __name__ == "__main__":
    main()
