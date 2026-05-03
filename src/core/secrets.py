import os


def get_gemini_api_key():
    if os.environ.get("GEMINI_API_KEY") is not None:
        return os.environ.get("GEMINI_API_KEY")
    elif os.path.exists(".secrets/GEMINI_API_KEY"):
        with open(".secrets/GEMINI_API_KEY", "r") as f:
            return f.read().strip()
    elif os.path.exists("../../.secrets/GEMINI_API_KEY"):
        with open("../../.secrets/GEMINI_API_KEY", "r") as f:
            return f.read().strip()
    else:
        raise ValueError("GEMINI_API_KEY not found")
