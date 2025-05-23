import time

import pandas as pd

S3_PARQUET_SOURCE_DIR = "s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model/"
S3_PARQUET_OUTPUT_PATH = "s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model.parquet"


def main():
    start_time = time.time()
    df = pd.read_parquet(S3_PARQUET_SOURCE_DIR)
    df.to_parquet(S3_PARQUET_OUTPUT_PATH)
    print(f"Took {(time.time() - start_time) / 60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
