import os

from tqdm import tqdm


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


def get_bkai_result_format(
    dataset: list[str | int], output_dir="predict.txt", stdin_out: bool = True
):
    for idx in tqdm(
        range(len(dataset)), desc="Start writing results into BKAI's format: ..."
    ):
        data = dataset[idx]
        result = " ".join(str(data))
        writer = open(output_dir, "a+")
        writer.writelines(result + "\n")
        if stdin_out:
            print(f"{idx}./ Writing {result}")
    print("Done")
