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
    yield df[i:i + batch_size]
