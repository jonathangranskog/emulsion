import time
from google import genai
from google.genai import types
from src.core.secrets import get_gemini_api_key


def get_gemini_client():
    return genai.Client(api_key=get_gemini_api_key())


def exponential_delay_embed_content(
    client: genai.Client,
    contents: str,
    model: str = "gemini-embedding-001",
    task_type: str = "SEMANTIC_SIMILARITY",
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
                    task_type=task_type,
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
