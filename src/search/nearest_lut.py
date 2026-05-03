import numpy as np
import os
import torch

from src.utils.gemini import exponential_delay_embed_content, get_gemini_client
from scipy.spatial.distance import cosine


class NearestLutSearch:
    def __init__(self, lut_cache_path: str = "assets/lut_cache.npy"):
        if not os.path.exists(lut_cache_path):
            print(f"LUT cache file does not exist: {lut_cache_path}")
            print("Falling back to tiny cache file...")
            lut_cache_path = "assets/tiny_lut_cache.npy"
            assert os.path.exists(lut_cache_path), (
                f"Tiny LUT cache file does not exist: {lut_cache_path}"
            )

        # Load the cache - it's a list of dictionaries
        self.lut_cache = np.load(lut_cache_path, allow_pickle=True)

        # Extract embeddings and LUTs from the cache
        self.lut_entries = []
        self.embeddings = []

        for entry in self.lut_cache:
            # Store the full entry for later retrieval
            self.lut_entries.append(entry)

            # Use short description embeddings for search
            self.embeddings.append(entry["short_embedding"])

        # Convert to numpy array for faster computation
        self.embeddings = np.array(self.embeddings)

        print(
            f"Loaded {len(self.lut_entries)} LUTs with {len(self.embeddings)} embeddings"
        )

        self.client = get_gemini_client()

    def search(self, query_string: str) -> dict:
        """
        Search for the nearest LUT based on a query string.

        Args:
            query_string: The text query to search for

         Returns:
            Dictionary containing the LUT data (lut_tensor, domain_min, domain_max, etc.)
        """
        results = self.search_top_k(query_string, k=1)
        return results[0]

    def search_top_k(self, query_string: str, k: int = 5) -> list[dict]:
        """
        Search for the top k nearest LUTs based on a query string.

        Args:
            query_string: The text query to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing LUT data, sorted by relevance
        """
        query_embedding = self.encode(query_string)

        # Calculate cosine distance to all embeddings
        distances = np.array([cosine(query_embedding, emb) for emb in self.embeddings])

        # Sort by distance and get top k
        top_indices = np.argsort(distances)[:k]

        # Build result list
        results = []
        for idx in top_indices:
            entry = dict(self.lut_entries[idx])
            entry["distance"] = distances[idx]
            results.append(entry)

        if results:
            print(
                f"Top match: {results[0]['lut_name']} (distance: {results[0]['distance']:.4f})"
            )

        return results

    def encode(self, query_string: str) -> np.ndarray:
        response = exponential_delay_embed_content(
            self.client,
            query_string,
            model="gemini-embedding-001",
        )
        return np.array(response.embeddings[0].values)


if __name__ == "__main__":
    search = NearestLutSearch()
    result = search.search("A dark and moody cinematic look")
    print(f"\nLUT tensor shape: {result['lut_tensor'].shape}")
    print(f"LUT name: {result['lut_name']}")
    print(f"LUT path: {result['lut_path']}")
    print(f"Domain min: {result['domain_min']}")
    print(f"Domain max: {result['domain_max']}")
