# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the UGround dataset to parquet format
"""

import argparse
import os
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


def read_parquet_file(file_path):
    """
    Read a parquet file containing viewport images and action data.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: DataFrame containing the data with columns:
            - viewport: Image bytes
            - vision_compatible_action: String describing the action
            - action_desc: String describing the action in more detail
    """
    df = pd.read_parquet(file_path)

    # Verify required columns exist
    required_columns = ["viewport", "vision_compatible_action", "action_desc"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def process_data(df, split):
    """
    Process the data into the required format.

    Args:
        df (pd.DataFrame): Input DataFrame
        split (str): Dataset split name ('train' or 'test')

    Returns:
        pd.DataFrame: Processed DataFrame
    """

    def process_fn(row, idx):
        data = {
            "data_source": "uground",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that map the action description to the action in the UI image. "
                        "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                        "The final answer MUST BE put in <answer> </answer> tags."
                    ),
                },
                {
                    "role": "user",
                    "content": row["action_desc"],
                },
            ],
            "images": [row["viewport"]],
            "ability": "vision",
            "reward_model": {
                "style": "rule",
                "ground_truth": row["vision_compatible_action"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": row["vision_compatible_action"],
                "question": row["action_desc"],
            },
        }
        return data

    processed_data = []
    for idx, row in df.iterrows():
        processed_data.append(process_fn(row, idx))

    return pd.DataFrame(processed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, help="Path to the input parquet file"
    )
    parser.add_argument("--local_dir", default="~/data/uground")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--split", default="train", choices=["train", "test"], help="Dataset split"
    )

    args = parser.parse_args()

    # Read the input parquet file
    df = read_parquet_file(args.input_file)

    # Process the data
    processed_df = process_data(df, args.split)

    # Save to local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    output_file = os.path.join(local_dir, f"{args.split}.parquet")
    processed_df.to_parquet(output_file)

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
