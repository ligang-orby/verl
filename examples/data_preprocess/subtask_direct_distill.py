"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse

import boto3
import ray


def get_s3_bucket_and_key_from_uri(s3_uri: str) -> tuple[str, str]:
    """
    Get the bucket and key from an S3 URI.

    Args:
        s3_uri (str): The S3 URI.

    Returns:
        tuple[str, str]: The bucket and key.
    """
    bucket, key = s3_uri.replace("s3://", "").strip().split("/", 1)
    return bucket, key


def list_s3_uris(s3_client, s3_uri: str) -> list[str]:
    """
    List all S3 URIs under the given S3 URI, original URI excluded.

    Args:
        s3_client (boto3.client): The S3 client.
        s3_uri (str): The S3 URI.

    Returns:
        list[str]: A list of S3 URIs.
    """
    bucket, prefix = get_s3_bucket_and_key_from_uri(s3_uri)
    s3_uris = []

    # Fetching all objects within the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Constructing the full S3 URI
            s3_uris.append(f"s3://{bucket}/{obj['Key']}")

    if s3_uri in s3_uris:
        s3_uris.remove(s3_uri)  # Remove the input URI from the list
    return s3_uris


@ray.remote
def data_processing_task(pb_uris_batch: list[str], output_path: str) -> None:
    """
    Process a batch of protobuf URIs and save the output to a parquet file.

    Args:
        pb_uris_batch (list[str]): A list of protobuf URIs.
        output_path (str): The output path.
    """
    pass


def main(input_path: str, output_path: str) -> None:
    s3_client = boto3.client("s3")
    pb_uris = list_s3_uris(s3_client, input_path)
    pb_uris_batches = [pb_uris[i : i + 100] for i in range(0, len(pb_uris), 100)]

    ray.init()
    ray.get([data_processing_task.remote(pb_uris_batch, output_path) for pb_uris_batch in pb_uris_batches])


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
