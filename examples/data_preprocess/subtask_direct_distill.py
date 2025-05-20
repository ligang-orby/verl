"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse

import boto3
import ray
from tqdm import tqdm

from .utils import s3_utils

ray.init()


@ray.remote
def data_processing_task(pb_uris_batch: list[str], output_path: str) -> int:
    """
    Process a batch of protobuf URIs and save the output to a parquet file.

    Args:
        pb_uris_batch (list[str]): A list of protobuf URIs.
        output_path (str): The output path to upload the parquet file.

    Returns:
        int: The number of data points created in the parquet file.
    """
    pass


def main(input_path: str, output_path: str) -> None:
    s3_client = boto3.client("s3")
    pb_uris = s3_utils.list_s3_uris(s3_client, input_path)
    pb_uris_batches = [pb_uris[i : i + 100] for i in range(0, len(pb_uris), 100)]

    tasks = [data_processing_task.remote(pb_uris_batch, output_path) for pb_uris_batch in pb_uris_batches]

    pbar = tqdm(total=len(tasks), desc="Processing batches")
    num_data_points = []
    while tasks:
        done_id, tasks = ray.wait(tasks, num_returns=1)
        result = ray.get(done_id[0])
        num_data_points.append(result)
        pbar.update(1)

    pbar.close()
    print(f"Total number of data points created: {sum(num_data_points)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", help="Path to the input trajectory protobuf S3 directory. Not necessary if split is provided.")
    parser.add_argument("--output_path", "-o", help="Path to the output parquet S3 directory. Not necessary if split is provided.")
    parser.add_argument("--split", "-s", help="Split name, can be 'train' or 'test'. If provided, uses the default train and test input and output paths for experiment 2.")
    args = parser.parse_args()

    if args.split:
        if args.input_path:
            raise ValueError("input_path should not be provided if split is provided.")
        if args.output_path:
            raise ValueError("output_path should not be provided if split is provided.")
        if args.split not in ["train", "test"]:
            raise ValueError("split should be 'train' or 'test'.")

        if args.split == "train":
            input_path = "s3://orby-osu-va/subtask/trajectories/experiment_2/train/filtered/"
            output_path = "s3://orby-osu-va/subtask/verl/experiment_2/train/"
        elif args.split == "test":
            input_path = "s3://orby-osu-va/subtask/trajectories/experiment_2/test/filtered/"
            output_path = "s3://orby-osu-va/subtask/verl/experiment_2/test/"
    else:
        if not args.input_path:
            raise ValueError("input_path should be provided if split is not provided.")
        if not args.output_path:
            raise ValueError("output_path should be provided if split is not provided.")

        input_path = args.input_path
        output_path = args.output_path

    main(input_path, output_path)
