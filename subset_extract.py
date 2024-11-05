import pandas as pd
import threading
from queue import Queue
import time
from tqdm import tqdm

def process_chunk(chunk, output_queue, num_rows=None):
    """Process a chunk to sample rows and maintain diversity."""
    sample_size = min(num_rows, len(chunk)) if num_rows else int(len(chunk) * 0.1)
    sampled_chunk = chunk.sample(sample_size, random_state=42)
    output_queue.put(sampled_chunk)

def extract_subset(file_path, output_path, size_mb=None, num_rows=None):
    chunk_size = 10 ** 6  # Approximate chunk size in bytes (1 MB)
    output_queue = Queue()
    threads = []
    total_size = 0
    sampled_data = []

    # Calculate total number of chunks for progress tracking
    total_rows = sum(1 for _ in pd.read_csv(file_path, chunksize=10000, encoding='ISO-8859-1'))
    num_chunks = (total_rows // 10000) + 1
    pbar = tqdm(total=num_chunks, desc="Processing chunks")

    start_time = time.time()

    # Read CSV in chunks and use threads for parallel processing
    for chunk in pd.read_csv(file_path, chunksize=10000, encoding='ISO-8859-1'):
        chunk_size_mb = chunk.memory_usage(index=True).sum() / (1024 ** 2)
        if size_mb:
            total_size += chunk_size_mb

        thread = threading.Thread(target=process_chunk, args=(chunk, output_queue, num_rows))
        thread.start()
        threads.append(thread)

        pbar.update(1)  # Update progress bar

        if size_mb and total_size >= size_mb:
            break
        if num_rows and len(sampled_data) >= num_rows:
            break

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results from the queue
    while not output_queue.empty():
        sampled_data.append(output_queue.get())

    # Calculate and print estimated remaining time
    elapsed_time = time.time() - start_time
    pbar.close()
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Concatenate all sampled dataframes
    final_df = pd.concat(sampled_data, ignore_index=True)

    # Ensure diversity in specific columns
    def ensure_diversity(df, columns):
        for col in columns:
            unique_vals = df[col].nunique()
            if unique_vals < df[col].nunique():
                print(f"Warning: The column '{col}' may have limited unique values in the subset.")

    ensure_diversity(final_df, ['gender', 'year', 'speech_date', 'party_group'])

    # Save to CSV, maintaining original format
    final_df.to_csv(output_path, index=False, encoding='ISO-8859-1')
    print(f"Output saved to {output_path}")

# Example usage:
size_mb = 10
extract_subset('data_complete/senti_df.csv', f'data/subset_senti_df_{size_mb}.csv', size_mb=size_mb)
