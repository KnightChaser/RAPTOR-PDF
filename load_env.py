# load_env.py
# Boilerplate code to load the environment variables from the .env file

import os
from dotenv import load_dotenv


def get_api_credentials() -> None:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if not openai_api_key and not openai_api_key.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY is not set in the environment variables")
