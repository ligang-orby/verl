"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse
import copy
import os
import re
import time
from typing import Literal, TypedDict

import boto3
import ray
from datasets import Dataset, Sequence
from datasets import Image as ImageData
from fm.action_data_pb2 import ActionData
from fm.llm_data_pb2 import LLMInteraction
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from orby.subtask.utils import action_parsing_utils, image_utils, s3_utils

VERL_IMAGE_TOKEN = "<image>\n"
DATA_SOURCE = "subtask_direct_distill"
GT_STYLE = "rule"
MAX_IMAGE_COUNT = 3

ray.init()
print(ray.available_resources())
try:
    os.environ.pop("RAY_ADDRESS")
except KeyError:
    pass


class PromptDict(TypedDict):
    role: str
    content: str


class RewardModelDict(TypedDict):
    style: str
    ground_truth: dict


class VERLDataPoint(TypedDict):
    data_source: str
    prompt: list[PromptDict]
    images: list[Image.Image]
    ability: str
    reward_model: RewardModelDict
    extra_info: dict


def get_llm_interaction_data(llm_interaction: LLMInteraction, ability: Literal["reward_model", "executor"]) -> tuple[str, str, list[Image.Image], dict]:
    """
    Get the system prompt, user prompt, images, and ground truth from a LLM interaction.
    """
    # System prompt
    assert llm_interaction.llm_messages[0].role == "system", "System prompt should be the first message"
    system_prompt = llm_interaction.llm_messages[0].llm_contents[0].text

    # User prompt
    assert llm_interaction.llm_messages[1].role == "user", "User prompt should be the second message"
    user_prompt_list = [llm_content.text if llm_content.text else VERL_IMAGE_TOKEN for llm_content in llm_interaction.llm_messages[1].llm_contents]
    # Keep only the last MAX_IMAGE_COUNT <image> tags due to Qwen-VL-2.5's max context length
    image_count = user_prompt_list.count(VERL_IMAGE_TOKEN)
    if image_count > MAX_IMAGE_COUNT:
        image_indices = [i for i, x in enumerate(user_prompt_list) if x == VERL_IMAGE_TOKEN]
        image_indices_to_remove = image_indices[:-MAX_IMAGE_COUNT]
        # Each action and thinking history is right under each image
        step_indices_to_remove = [i + 1 for i in image_indices_to_remove]
        indices_to_remove = sorted(set(image_indices_to_remove + step_indices_to_remove))

        # Heuristically, we need to remove the first "Step 1:" and replace it with the first step not removed
        first_step_not_removed = len(image_indices_to_remove) + 1
        user_prompt_list[0] = user_prompt_list[0].replace("Step 1:\n", f"Step {first_step_not_removed}:\n")

        for idx in reversed(indices_to_remove):
            user_prompt_list.pop(idx)
    user_prompt = "".join(user_prompt_list)

    # Images
    image_urls = [llm_content.image_url for llm_content in llm_interaction.llm_messages[1].llm_contents if llm_content.image_url]
    # Keep only the last MAX_IMAGE_COUNT images due to Qwen-VL-2.5's max context length
    if len(image_urls) > MAX_IMAGE_COUNT:
        image_urls = image_urls[-MAX_IMAGE_COUNT:]
    images = [image_utils.convert_image_to_pil_image(image_url) for image_url in image_urls]

    # Ground truth
    if ability == "reward_model":
        ground_truth = action_parsing_utils.extract_content_by_tags(llm_interaction.response, ["reasoning", "should_end", "goal_achieved", "answer"])
    elif ability == "executor":
        ground_truth = action_parsing_utils.extract_content_by_tags(llm_interaction.response, ["thinking", "action"])
    else:
        raise ValueError(f"Invalid ability: {ability}")
    for key in ground_truth.keys():
        ground_truth[key] = ground_truth[key].strip()

    return system_prompt, user_prompt, images, ground_truth


def convert_action_to_datapoints(action: ActionData, step_idx: int) -> list[VERLDataPoint]:
    """
    Convert individual action to a VERLDataPoint.

    Args:
        action (ActionData): The action to convert.

    Returns:
        VERLDataPoint: The converted data point.
    """

    llm_interactions = action.agent_state.llm_interactions
    assert len(llm_interactions) > 0, "No LLM interactions found"
    assert len(llm_interactions) <= 2, "More than 2 LLM interactions found"

    reward_model_interaction = llm_interactions[0]
    ability = "reward_model"

    (
        reward_model_system_prompt,
        reward_model_user_prompt,
        reward_model_images,
        reward_model_ground_truth,
    ) = get_llm_interaction_data(reward_model_interaction, ability)

    extra_info = {
        "action_id": action.id,
        "step_idx": step_idx,
    }

    reward_model_data_point = VERLDataPoint(
        data_source=DATA_SOURCE,
        prompt=[
            PromptDict(
                role="user",  # Note: this is NOT a mistake. We use user prompts for both to conform with vanilla Qwen-VL-2.5
                content=reward_model_system_prompt,
            ),
            PromptDict(
                role="user",
                content=reward_model_user_prompt,
            ),
        ],
        images=reward_model_images,
        ability=ability,
        reward_model=RewardModelDict(
            style=GT_STYLE,
            ground_truth=reward_model_ground_truth,
        ),
        extra_info=extra_info,
    )

    executor_data_point = None
    if len(llm_interactions) == 2:
        ability = "executor"
        executor_interaction = llm_interactions[1]

        (
            executor_system_prompt,
            executor_user_prompt,
            executor_images,
            executor_ground_truth,
        ) = get_llm_interaction_data(executor_interaction, ability)

        executor_data_point = VERLDataPoint(
            data_source=DATA_SOURCE,
            prompt=[
                PromptDict(
                    role="user",
                    content=executor_system_prompt,  # Note: this is NOT a mistake. We use user prompts for both to conform with vanilla Qwen-VL-2.5
                ),
                PromptDict(
                    role="user",
                    content=executor_user_prompt,
                ),
            ],
            images=executor_images,
            ability=ability,
            reward_model=RewardModelDict(
                style=GT_STYLE,
                ground_truth=executor_ground_truth,
            ),
            extra_info=extra_info,
        )

    return [reward_model_data_point, executor_data_point]


@ray.remote
def data_processing_task(pb_uris_batch: list[str], batch_idx: int, output_path: str, filter_overlong_prompts: bool = False, max_prompt_length: int = 1024, processor=None, tokenizer=None) -> dict:
    """
    Process a batch of protobuf URIs and save the output to a parquet file.

    Args:
        pb_uris_batch (list[str]): A list of protobuf URIs.
        output_path (str): The output path to upload the parquet file.
        filter_overlong_prompts (bool): Whether to filter out overlong prompts.
        max_prompt_length (int): Maximum prompt length for filtering.
        processor: HF model processor for long prompt filtering.
        tokenizer: HF model tokenizer for long prompt filtering.

    Returns:
        dict: Statistics about data processing and filtering.
    """
    reward_model_data_list: list[VERLDataPoint] = []
    executor_data_list: list[VERLDataPoint] = []

    # Statistics tracking
    stats = {
        "total_reward_model": 0,
        "total_executor": 0,
        "filtered_reward_model": 0,
        "filtered_executor": 0,
        "kept_reward_model": 0,
        "kept_executor": 0,
    }

    for pb_uri in pb_uris_batch:
        td = s3_utils.load_trajectory_data_from_s3(pb_uri)

        for idx, action in enumerate(td.actions):
            data_points = convert_action_to_datapoints(action, idx)

            # Process reward model data point
            reward_model_dp = data_points[0]
            stats["total_reward_model"] += 1

            if filter_overlong_prompts and tokenizer:
                if should_filter_datapoint(reward_model_dp, processor, tokenizer, max_prompt_length):
                    stats["filtered_reward_model"] += 1
                else:
                    reward_model_data_list.append(reward_model_dp)
                    stats["kept_reward_model"] += 1
            else:
                reward_model_data_list.append(reward_model_dp)
                stats["kept_reward_model"] += 1

            # Process executor data point if it exists
            if data_points[1]:
                executor_dp = data_points[1]
                stats["total_executor"] += 1

                if filter_overlong_prompts and tokenizer:
                    if should_filter_datapoint(executor_dp, processor, tokenizer, max_prompt_length):
                        stats["filtered_executor"] += 1
                    else:
                        executor_data_list.append(executor_dp)
                        stats["kept_executor"] += 1
                else:
                    executor_data_list.append(executor_dp)
                    stats["kept_executor"] += 1

    # Only create datasets and save if we have data
    if reward_model_data_list:
        reward_model_dataset = Dataset.from_list(reward_model_data_list)
        reward_model_dataset = reward_model_dataset.cast_column("images", Sequence(ImageData()))
        reward_model_output_path = f"{output_path.rstrip('/')}/reward_model/reward_model_{batch_idx:04d}.parquet"
        reward_model_dataset.to_parquet(reward_model_output_path)

    if executor_data_list:
        executor_dataset = Dataset.from_list(executor_data_list)
        executor_dataset = executor_dataset.cast_column("images", Sequence(ImageData()))
        executor_output_path = f"{output_path.rstrip('/')}/executor/executor_{batch_idx:04d}.parquet"
        executor_dataset.to_parquet(executor_output_path)

    return stats


def main(input_path: str, output_path: str, filter_overlong_prompts: bool = False, max_prompt_length: int = 1024, model_name: str = None) -> None:
    print("Start data processing with parameters:")
    print(f"- Input path: {input_path}")
    print(f"- Output path: {output_path}")
    print(f"- Filter overlong prompts: {filter_overlong_prompts}")
    print(f"- Max prompt length: {max_prompt_length}")
    print(f"- Model name: {model_name}")

    start_time = time.time()
    s3_client = boto3.client("s3")
    pb_uris = s3_utils.list_s3_uris(s3_client, input_path)
    pb_uris_batches = [pb_uris[i : i + 10] for i in range(0, len(pb_uris), 10)]

    # Load processor and tokenizer once if filtering is enabled
    processor = None
    tokenizer = None
    if filter_overlong_prompts and model_name:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded processor and tokenizer for model {model_name}.")

    tasks = [
        data_processing_task.remote(
            pb_uris_batch,
            batch_idx,
            output_path,
            filter_overlong_prompts,
            max_prompt_length,
            processor,
            tokenizer,
        )
        for batch_idx, pb_uris_batch in enumerate(pb_uris_batches)
    ]

    pbar = tqdm(total=len(tasks), desc="Processing batches")
    all_stats = []
    while tasks:
        done_id, tasks = ray.wait(tasks, num_returns=1)
        result = ray.get(done_id[0])
        all_stats.append(result)
        pbar.update(1)

    pbar.close()

    # Aggregate statistics
    total_stats = {
        "total_reward_model": sum(s["total_reward_model"] for s in all_stats),
        "total_executor": sum(s["total_executor"] for s in all_stats),
        "filtered_reward_model": sum(s["filtered_reward_model"] for s in all_stats),
        "filtered_executor": sum(s["filtered_executor"] for s in all_stats),
        "kept_reward_model": sum(s["kept_reward_model"] for s in all_stats),
        "kept_executor": sum(s["kept_executor"] for s in all_stats),
    }

    # Print statistics
    print("\n" + "=" * 60)
    print("DATA PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total reward model data points: {total_stats['total_reward_model']}")
    print(f"Total executor data points: {total_stats['total_executor']}")
    print(f"Total data points: {total_stats['total_reward_model'] + total_stats['total_executor']}")

    if filter_overlong_prompts:
        print(f"\nFILTERING STATISTICS (max_prompt_length={max_prompt_length}):")
        print(f"Filtered reward model data points: {total_stats['filtered_reward_model']} ({total_stats['filtered_reward_model'] / total_stats['total_reward_model'] * 100:.1f}%)")
        print(f"Filtered executor data points: {total_stats['filtered_executor']} ({total_stats['filtered_executor'] / total_stats['total_executor'] * 100:.1f}%)")
        print(f"Total filtered: {total_stats['filtered_reward_model'] + total_stats['filtered_executor']} ({(total_stats['filtered_reward_model'] + total_stats['filtered_executor']) / (total_stats['total_reward_model'] + total_stats['total_executor']) * 100:.1f}%)")

    print(f"\nKept reward model data points: {total_stats['kept_reward_model']}")
    print(f"Kept executor data points: {total_stats['kept_executor']}")
    print(f"Total kept data points: {total_stats['kept_reward_model'] + total_stats['kept_executor']}")

    print(f"\nProcessing took {(time.time() - start_time) / 60:.1f} minutes")
    print("Done!")
    print("=" * 60)


def should_filter_datapoint(data_point: VERLDataPoint, processor, tokenizer, max_prompt_length: int) -> bool:
    """
    Check if a data point should be filtered out based on token length.

    Args:
        data_point: The data point to check
        processor: The processor for multimodal tokenization
        tokenizer: The tokenizer for text-only tokenization
        max_prompt_length: Maximum allowed prompt length

    Returns:
        True if the data point should be filtered out (too long), False otherwise
    """
    try:
        messages = copy.deepcopy(data_point["prompt"])
        images = data_point.get("images", [])

        # If we have images and a processor, use multimodal tokenization
        if processor is not None and images:
            # Process messages to handle image tokens (similar to _build_messages in rl_dataset.py)
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            # Apply chat template and tokenize with images
            raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = processor(text=[raw_prompt], images=images, return_tensors="pt")

            # Get actual token length including images
            actual_length = model_inputs["input_ids"].shape[1]

            return actual_length > max_prompt_length
        else:
            # Text-only tokenization
            text_length = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            return text_length > max_prompt_length

    except Exception as e:
        print(f"Warning: Failed to tokenize data point for filtering, keeping it: {e}")
        return False  # Keep the data point if we can't tokenize it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", help="Path to the input trajectory protobuf S3 directory. Not necessary if split is provided.")
    parser.add_argument("--output_path", "-o", help="Path to the output parquet S3 directory. Not necessary if split is provided.")
    parser.add_argument("--split", "-s", help="Split name, can be 'train' or 'test'. If provided, uses the default train and test input and output paths for experiment 2.")
    parser.add_argument("--max_prompt_length", type=int, default=7680, help="Maximum prompt length in tokens (default: 1024)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name for tokenizer/processor (default: Qwen/Qwen2.5-VL-7B-Instruct)")
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

    filter_overlong_prompts = True
    max_prompt_length = args.max_prompt_length
    model_name = args.model_name

    main(input_path, output_path, filter_overlong_prompts, max_prompt_length, model_name)
