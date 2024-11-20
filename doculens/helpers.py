import ast
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def setup_env_var(env_name: str):
    try:
        os.environ[env_name] = os.getenv(env_name)
    except:
        raise ImportError(f"Can not find {env_name} in .env file.")


# This script is highly recommended for individuals that have limit computatation resources.
def process_data_in_batches(df, batch_size=10000):
    """Processes data in batches.

    Args:
      df: The Pandas DataFrame to process.
      batch_size: The number of rows to process in each batch.

    Yields:
      A generator that yields batches of the DataFrame.
    """
    for i in range(0, len(df), batch_size):
        yield df[i : i + batch_size]


def get_env(env_name: str, default_name: str) -> str:
    "Get variables from the environment, if there is not, return default value"
    return os.getenv(env_name) if os.getenv(env_name) else default_name
