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
Preprocess the Screenspot dataset to parquet format
"""

import argparse
import io
import os
import json
import logging

import pandas as pd
from PIL import Image
from datasets import Dataset, Sequence
from datasets import Image as ImageData

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)


def read_parquet_file(file_path):
    """
    Read a parquet file containing viewport images and action data.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: DataFrame containing the data with columns:
            - viewport: Image bytes
            - instruction: String describing the action
            - bbox: List of bounding box coordinates [x1, y1, x2, y2]
    """
    df = pd.read_parquet(file_path)

    # Verify required columns exist
    required_columns = [
        "image",
        "instruction",
        "bbox",
    ]
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
        # Get image and resize ratios
        image = Image.open(io.BytesIO(row["image"]))
        
        # Get bounding box between 0 and 1
        ground_truth = {
            "bbox": row["bbox"],
        }

        instruction = row["instruction"].strip()

        data = {
            "data_source": "screenspot",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Map the user instruction to the coordinates in the UI image. "
                        "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                        "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                        "<image> Instruction: " + instruction
                    ),
                },
            ],
            "images": [image],
            "ability": "vision",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": instruction,
                "bounding_box": row["bbox"],
            },
        }
        return data

    for idx, row in df.iterrows():
        data = process_fn(row, idx)
        if len(data["prompt"][0]["content"]) > 4000:
            logging.warning(
                "Too long prompt: " + str(len(data["prompt"][0]["content"]))
            )
            # Filter out the data with too long prompt
            continue
        yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, help="Path to the input parquet file"
    )
    parser.add_argument("--local_dir", default="~/data/screenspot")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "dev"],
        help="Dataset split",
    )

    args = parser.parse_args()

    # Save to local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    output_file = os.path.join(local_dir, f"{args.split}.parquet")

    # Read the input parquet file and save it to dataset.
    df = read_parquet_file(args.input_file)
    # Have to do this for the PIL Image object, otherwise causing conversion type error.
    dataset = Dataset.from_generator(
        process_data, gen_kwargs={"df": df, "split": args.split}
    )
    dataset = dataset.cast_column("images", Sequence(ImageData()))
    dataset.to_parquet(output_file)

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir) 