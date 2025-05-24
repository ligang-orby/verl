import time

import pandas as pd

S3_PARQUET_PAIRS = [
    ["s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model/", "s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model.parquet"],
    ["s3://orby-osu-va/subtask/verl/experiment_2/test/executor/", "s3://orby-osu-va/subtask/verl/experiment_2/test/executor.parquet"],
    ["s3://orby-osu-va/subtask/verl/experiment_2/train/reward_model/", "s3://orby-osu-va/subtask/verl/experiment_2/train/reward_model.parquet"],
    ["s3://orby-osu-va/subtask/verl/experiment_2/train/executor/", "s3://orby-osu-va/subtask/verl/experiment_2/train/executor.parquet"],
]


def main():
    for source_dir, output_path in S3_PARQUET_PAIRS:
        start_time = time.time()
        df = pd.read_parquet(source_dir)
        df.to_parquet(output_path)
        print(f"${source_dir} took {(time.time() - start_time) / 60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
