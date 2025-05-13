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
Reward scoring for UI UGround task
"""

import re
from typing import Dict, List, Tuple

from verl.utils.reward_score.base import BaseRewardScorer


class UIGroundRewardScorer(BaseRewardScorer):
    """Reward scorer for UI UGround task."""

    def __init__(self):
        super().__init__()
        self.thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self.action_pattern = re.compile(r"(\w+)\((\d+),\s*(\d+)\)")

    def _extract_action_info(self, action_str: str) -> Tuple[str, int, int]:
        """Extract action type and coordinates from action string.

        Args:
            action_str: Action string in format "action_type(x, y)"

        Returns:
            Tuple of (action_type, x, y)
        """
        match = self.action_pattern.match(action_str.strip())
        if not match:
            return "", 0, 0
        action_type, x, y = match.groups()
        return action_type, int(x), int(y)

    def _check_coordinates_in_bbox(
        self, x: int, y: int, bbox: List[int], tolerance: int = 5
    ) -> bool:
        """Check if coordinates are within bounding box with tolerance.

        Args:
            x: X coordinate
            y: Y coordinate
            bbox: Bounding box [x1, y1, x2, y2]
            tolerance: Pixel tolerance for coordinate matching

        Returns:
            True if coordinates are within bbox with tolerance
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 - tolerance <= x <= x2 + tolerance
            and y1 - tolerance <= y <= y2 + tolerance
        )

    def score(self, prediction: str, ground_truth: Dict) -> Dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information
                - action: Ground truth action string
                - bbox: Optional bounding box [x1, y1, x2, y2]

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - details: Dictionary with individual check results
        """
        # Check 1: Format validation
        has_thinking = bool(self.thinking_pattern.search(prediction))
        answer_match = self.answer_pattern.search(prediction)
        has_answer = bool(answer_match)

        # Check 2: Action type validation
        gt_action_type, _, _ = self._extract_action_info(ground_truth["action"])
        pred_answer = answer_match.group(1).strip() if answer_match else ""
        pred_action_type, pred_x, pred_y = self._extract_action_info(pred_answer)
        action_type_correct = pred_action_type == gt_action_type

        # Check 3: Coordinate validation
        bbox = ground_truth.get("bbox")
        coordinates_correct = False
        if bbox is not None:
            coordinates_correct = self._check_coordinates_in_bbox(pred_x, pred_y, bbox)

        # Calculate overall score
        format_score = 1.0 if (has_thinking and has_answer) else 0.0
        action_score = 1.0 if action_type_correct else 0.0
        coord_score = 1.0 if coordinates_correct else 0.0

        # Weight the scores (can be adjusted based on importance)
        weights = {"format": 0.2, "action_type": 0.3, "coordinates": 0.5}

        overall_score = (
            weights["format"] * format_score
            + weights["action_type"] * action_score
            + weights["coordinates"] * coord_score
        )

        return {
            "score": overall_score,
            "details": {
                "format_check": {
                    "has_thinking": has_thinking,
                    "has_answer": has_answer,
                    "score": format_score,
                },
                "action_type_check": {
                    "predicted": pred_action_type,
                    "ground_truth": gt_action_type,
                    "score": action_score,
                },
                "coordinate_check": {
                    "predicted": (pred_x, pred_y),
                    "ground_truth_bbox": bbox,
                    "score": coord_score,
                },
            },
        }


def compute_score(prediction: str, ground_truth: Dict) -> Dict:
    """Compute score for a single prediction.

    Args:
        prediction: Prediction string
        ground_truth: Dictionary containing ground truth information
            - action: Ground truth action string
            - bbox: Optional bounding box [x1, y1, x2, y2]

    Returns:
        Dictionary containing:
            - score: Overall score (0-1)
            - details: Dictionary with individual check results
    """
    scorer = UIGroundRewardScorer()
    return scorer.score(prediction, ground_truth)
