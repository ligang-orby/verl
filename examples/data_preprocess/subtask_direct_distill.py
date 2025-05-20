"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse
from typing import TypedDict

import boto3
import pandas as pd
import ray
from fm.action_data_pb2 import ActionData
from PIL import Image
from tqdm import tqdm

from .utils import s3_utils

ray.init()


class PromptDict(TypedDict):
    role: str
    content: str


class RewardModelDict(TypedDict):
    style: str = "rule"
    ground_truth: dict


class VERLDataPoint(TypedDict):
    data_source: str = "subtask_direct_distill"
    prompt: list[PromptDict]
    images: list[Image.Image]
    ability: str = "vision"
    reward_model: RewardModelDict
    extra_info: dict


def convert_action_to_datapoint(action: ActionData) -> VERLDataPoint:
    """
    Convert individual action to a VERLDataPoint.

    Args:
        action (ActionData): The action to convert.

    Returns:
        VERLDataPoint: The converted data point.
    """
    pass


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
    for pb_uri in pb_uris_batch:
        td = s3_utils.load_trajectory_data_from_s3(pb_uri)

        data_list: list[VERLDataPoint] = []
        for action in td.actions:
            data = convert_action_to_datapoint(action)
            data_list.append(data)

        data_df = pd.DataFrame(data_list)
        data_df.to_parquet(output_path, index=False)
        return len(data_list)


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
