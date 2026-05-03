import json
import os
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from src.core.secrets import get_gemini_api_key
from src.utils.lut_utils import read_cube_file


def exponential_delay_embed_content(
    client: genai.Client,
    contents: str,
    model: str = "gemini-embedding-001",
    max_retries: int = 10,
    initial_delay: float = 10.0,
    max_delay: float = 1800.0,
):
    """
    Call Gemini's embed_content with exponential backoff for rate limiting.

    Args:
        client: The Gemini client instance
        contents: The content to embed
        model: The model name to use
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds

    Returns:
        The embedding response from Gemini

    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                ),
            )
            return response
        except Exception as e:
            # Check if it's a rate limit error
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

            if not is_rate_limit or attempt == max_retries - 1:
                # If it's not a rate limit error or we're out of retries, raise
                print(f"❌ Error calling Gemini API: {e}")
                raise

            # Wait with exponential backoff
            wait_time = min(delay, max_delay)
            print(
                f"⚠️  Rate limit hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
            delay *= 2  # Exponential increase

    raise Exception("Max retries exhausted")


def create_lut_embeddings(captions_dir: str, output_dir: str):
    """
    Process all LUT JSON files in the captions directory and create embeddings.

    Args:
        captions_dir: Directory containing LUT JSON files
        output_dir: Directory to save the embeddings and LUT data
    """
    # Initialize Gemini client
    client = genai.Client(api_key=get_gemini_api_key())

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the captions directory
    captions_path = Path(captions_dir)
    json_files = list(captions_path.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        # Check if embedding already exists
        output_filename = json_file.stem + ".npy"
        output_file = output_path / output_filename

        if output_file.exists():
            print(f"Skipping {json_file.name} (embedding already exists)")
            continue

        print(f"\nProcessing: {json_file.name}")

        try:
            # Load JSON file
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract descriptions
            short_description = data.get("short_description", "")
            long_description = data.get("long_description", "")
            lut_path = data.get("lut_path", "")

            if not lut_path:
                print(f"  Warning: No lut_path found in {json_file.name}")
                continue

            # Load LUT tensor
            lut_tensor, domain_min, domain_max = read_cube_file(lut_path)
            print(f"  Loaded LUT: {lut_path} (shape: {lut_tensor.shape})")

            # Create embeddings for both descriptions
            print("  Computing embeddings...")

            short_embedding = None
            long_embedding = None

            if short_description:
                short_result = exponential_delay_embed_content(
                    client=client,
                    contents=short_description,
                    model="gemini-embedding-001",
                )
                short_embedding = np.array(short_result.embeddings[0].values)
                print(f"    Short description embedding: {short_embedding.shape}")

            if long_description:
                long_result = exponential_delay_embed_content(
                    client=client,
                    contents=long_description,
                    model="gemini-embedding-001",
                )
                long_embedding = np.array(long_result.embeddings[0].values)
                print(f"    Long description embedding: {long_embedding.shape}")

            # Prepare data to save
            output_data = {
                "lut_tensor": lut_tensor.numpy(),
                "domain_min": domain_min,
                "domain_max": domain_max,
                "short_description": short_description,
                "long_description": long_description,
                "short_embedding": short_embedding,
                "long_embedding": long_embedding,
                "lut_path": lut_path,
            }

            # Save to npy file (use same name as JSON file)
            np.save(output_file, output_data, allow_pickle=True)
            print(f"  Saved to: {output_file}")

        except Exception as e:
            print(f"  Error processing {json_file.name}: {str(e)}")
            continue

    print(f"\n✓ Processing complete! Embeddings saved to {output_dir}")


if __name__ == "__main__":
    # Set paths
    captions_dir = "assets/captions"
    output_dir = "assets/lut_embeddings"

    create_lut_embeddings(captions_dir, output_dir)
