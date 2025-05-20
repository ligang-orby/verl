"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse


def main(input_path: str, output_path: str) -> None:
    pass


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
