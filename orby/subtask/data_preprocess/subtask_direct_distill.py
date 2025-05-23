"""
Preprocess the subtask SVA v3 dataset to parquet format
"""

import argparse
import os
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

from orby.subtask.utils import action_parsing_utils, image_utils, s3_utils

VERL_IMAGE_TOKEN = "<image>\n"
DATA_SOURCE = "subtask_direct_distill"
GT_STYLE = "rule"

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
    # Keep only the last 4 <image> tags due to Qwen-VL-2.5's max context length
    image_count = user_prompt_list.count(VERL_IMAGE_TOKEN)
    if image_count > 4:
        image_indices = [i for i, x in enumerate(user_prompt_list) if x == VERL_IMAGE_TOKEN]
        indices_to_remove = image_indices[:-4]
        for idx in reversed(indices_to_remove):
            user_prompt_list.pop(idx)
    user_prompt = "".join(user_prompt_list)

    # Images
    image_urls = [llm_content.image_url for llm_content in llm_interaction.llm_messages[1].llm_contents if llm_content.image_url]
    # Keep only the last 4 images due to Qwen-VL-2.5's max context length
    if len(image_urls) > 4:
        image_urls = image_urls[-4:]
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


def convert_action_to_datapoints(action: ActionData, step_idx: int) -> VERLDataPoint:
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
def data_processing_task(pb_uris_batch: list[str], batch_idx: int, output_path: str) -> int:
    """
    Process a batch of protobuf URIs and save the output to a parquet file.

    Args:
        pb_uris_batch (list[str]): A list of protobuf URIs.
        output_path (str): The output path to upload the parquet file.

    Returns:
        int: The number of data points created in the parquet file.
    """
    reward_model_data_list: list[VERLDataPoint] = []
    executor_data_list: list[VERLDataPoint] = []

    for pb_uri in pb_uris_batch:
        td = s3_utils.load_trajectory_data_from_s3(pb_uri)

        for idx, action in enumerate(td.actions):
            data_points = convert_action_to_datapoints(action, idx)
            reward_model_data_list.append(data_points[0])
            if data_points[1]:
                executor_data_list.append(data_points[1])

    reward_model_dataset = Dataset.from_list(reward_model_data_list)
    executor_dataset = Dataset.from_list(executor_data_list)
    reward_model_dataset = reward_model_dataset.cast_column("images", Sequence(ImageData()))
    executor_dataset = executor_dataset.cast_column("images", Sequence(ImageData()))

    reward_model_output_path = f"{output_path.rstrip('/')}/reward_model/reward_model_{batch_idx:04d}.parquet"
    executor_output_path = f"{output_path.rstrip('/')}/executor/executor_{batch_idx:04d}.parquet"
    reward_model_dataset.to_parquet(reward_model_output_path)
    executor_dataset.to_parquet(executor_output_path)

    return len(reward_model_data_list) + len(executor_data_list)


def main(input_path: str, output_path: str) -> None:
    start_time = time.time()
    s3_client = boto3.client("s3")
    pb_uris = s3_utils.list_s3_uris(s3_client, input_path)
    pb_uris_batches = [pb_uris[i : i + 10] for i in range(0, len(pb_uris), 10)]

    tasks = [data_processing_task.remote(pb_uris_batch, batch_idx, output_path) for batch_idx, pb_uris_batch in enumerate(pb_uris_batches)]

    pbar = tqdm(total=len(tasks), desc="Processing batches")
    num_data_points = []
    while tasks:
        done_id, tasks = ray.wait(tasks, num_returns=1)
        result = ray.get(done_id[0])
        num_data_points.append(result)
        pbar.update(1)

    pbar.close()

    print(f"Total number of data points (reward model + executor) created: {sum(num_data_points)}")
    print(f"Took {(time.time() - start_time) / 60:.1f} minutes")
    print("Done!")


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
