import os
import ast
from dotenv import load_dotenv

import pandas as pd

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


def get_bkai_result_format(input_csv_path, output_txt_path):
    "Return results data from pandas into the format: qid cid1 cid2 ..."
    # Load data
    result_df = pd.read_csv(input_csv_path, index_col=False)

    # Parse 'cid' strings to lists
    result_df["cid"] = result_df["cid"].apply(ast.literal_eval)

    # Build the output lines
    lines = [
        f"{q} " + " ".join(map(str, cid)) + "\n"
        for q, cid in zip(result_df["qid"], result_df["cid"])
    ]

    # Write all lines to file at once
    with open(output_txt_path, "w") as file:
        file.writelines(lines)
