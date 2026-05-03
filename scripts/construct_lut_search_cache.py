"""
Takes a set of npy files containing LUTs and embeddings, and constructs
a cache of each LUT and their corresponding embeddings. The cache is just another
npy file that contains a list of LUT tensors and their corresponding embeddings.
"""

from pathlib import Path

import numpy as np


def construct_lut_search_cache(embeddings_dir: str, output_file: str):
    """
    Read all individual LUT embedding npy files and combine them into a single cache file.

    Args:
        embeddings_dir: Directory containing individual .npy files with LUT data
        output_file: Output path for the combined cache file
    """
    embeddings_path = Path(embeddings_dir)

    if not embeddings_path.exists():
        print(f"❌ Error: Directory {embeddings_dir} does not exist")
        return

    # Get all npy files
    npy_files = sorted(list(embeddings_path.glob("*.npy")))

    if not npy_files:
        print(f"❌ Error: No .npy files found in {embeddings_dir}")
        return

    print(f"Found {len(npy_files)} .npy files to process")

    # List to store all LUT data
    cache_data = []

    for npy_file in npy_files:
        print(f"Processing: {npy_file.name}")

        try:
            # Load the npy file
            data = np.load(npy_file, allow_pickle=True).item()

            # Extract the required fields
            lut_tensor = data.get("lut_tensor")
            short_embedding = data.get("short_embedding")
            long_embedding = data.get("long_embedding")
            domain_min = data.get("domain_min", [0.0, 0.0, 0.0])
            domain_max = data.get("domain_max", [1.0, 1.0, 1.0])
            lut_path = data.get("lut_path", "")

            # Validate that we have all required data
            if lut_tensor is None:
                print(f"  ⚠️  Warning: No lut_tensor in {npy_file.name}, skipping")
                continue

            if short_embedding is None or long_embedding is None:
                print(f"  ⚠️  Warning: Missing embeddings in {npy_file.name}, skipping")
                continue

            # Create cache entry
            cache_entry = {
                "lut_tensor": lut_tensor,
                "domain_min": domain_min,
                "domain_max": domain_max,
                "short_embedding": short_embedding,
                "long_embedding": long_embedding,
                "lut_path": lut_path,
                "lut_name": npy_file.stem,  # Store the name for reference
            }

            cache_data.append(cache_entry)
            print(
                f"  ✓ Added to cache (LUT shape: {lut_tensor.shape}, "
                f"Short embedding: {short_embedding.shape}, "
                f"Long embedding: {long_embedding.shape})"
            )

        except Exception as e:
            print(f"  ❌ Error processing {npy_file.name}: {str(e)}")
            continue

    if not cache_data:
        print("❌ Error: No valid data to save")
        return

    # Save the combined cache
    print(f"\nSaving {len(cache_data)} LUT entries to {output_file}...")
    np.save(output_file, cache_data, allow_pickle=True)
    print(f"✓ Cache saved successfully!")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total LUTs in cache: {len(cache_data)}")
    print(f"  Cache file: {output_file}")
    print(
        f"  Cache file size: {Path(output_file).stat().st_size / (1024 * 1024):.2f} MB"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set paths
    embeddings_dir = "assets/lut_embeddings"
    output_file = "assets/lut_cache.npy"

    construct_lut_search_cache(embeddings_dir, output_file)
