"""
This script is the class definition for the online verifier for ordering task.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
import numpy as np
import json
import re
import ast
from collections import defaultdict

from EmbodiedVLM.utils.qa_gen_utils import TaskData
from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
from EmbodiedVLM.utils.scene_graph_utils import SceneGraphReader


class TaskWiseOrderingVerifier:
    """
    This class is the class definition for the online verifier for ordering task.
    Given the shuffled frame id sequence, read the correct sequence idx, read the input sequence idx, read the task name,
    verfier if the input sequence is correct.
    An input sequence is correct if:
    1. The input sequence is the same as the correct sequence.
    2. The input sequence can reflect the correct sequence, that is, the diffs of the correct sequence is an subset of the diffs of the input sequence.
    3. The input sequence has high similarity with the correct sequence.
    """
    
    def __init__(
        self,
        task_data: TaskData
    ):
        """
        Initialize TaskWiseOrderingVerifier with pre-created TaskData.
        
        Args:
            task_data (TaskData): Pre-configured TaskData object
        """
        self.task_data = task_data
        self.task_name = task_data.task_name
        self.translator = StateChangeTranslator(type='forward_dynamics')
        
        print(f"Loaded task: {self.task_name} with {len(task_data.key_frame_ids)} key frames")

    @property
    def sensor_names(self) -> List[str]:
        """
        Extract sensor names from the image_paths dictionary.
        """
        if not self.task_data.image_paths:
            return []
        # Get sensor names from the first frame's image paths
        first_frame_id = list(self.task_data.image_paths.keys())[0]
        return list(self.task_data.image_paths[first_frame_id].keys())

    def _translate_sequence_to_signatures(
        self,
        sequence: List[str],
        partial_diff: bool = True
    ) -> List[Set[str]]:
        """
        Translate a sequence of frame IDs into a sequence of signatures.
        """
        signatures = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = self.task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=partial_diff
            )
            signature = self.translator.translate_diff_into_signatures(diff)
            signatures.append(signature)
        return signatures


    
    def _calculate_pair_correct_ratio_with_alignment(
        self,
        correct_signatures: List[Set[str]],
        input_signatures: List[Set[str]],
        correct_full_signatures: List[Set[str]] = None
    ) -> float:
        """
        Calculate pair correct ratio using optimal alignment between sequences of different lengths.
        Uses dynamic programming to find the best alignment that maximizes correct pairs.
        
        Args:
            correct_signatures: Ground truth signatures (partial diff)
            input_signatures: Input signatures (partial or full diff)
            correct_full_signatures: Ground truth full signatures (for inverse dynamics)
            
        Returns:
            Ratio of correctly aligned pairs (0 to 1)
        """
        if not correct_signatures or not input_signatures:
            return 0.0
            
        m, n = len(correct_signatures), len(input_signatures)
        
        # Use full signatures for comparison if provided (inverse dynamics)
        comparison_signatures = correct_full_signatures if correct_full_signatures else correct_signatures
        
        # DP table to track maximum correct pairs
        # dp[i][j] = max correct pairs using first i correct signatures and first j input signatures
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Option 1: Don't align current signatures
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                # Option 2: Align current signatures if they match
                if correct_signatures[i-1].issubset(input_signatures[j-1]):
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
                elif correct_full_signatures and input_signatures[j-1].issubset(comparison_signatures[i-1]):
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
        
        # The maximum possible pairs is the minimum of the two sequence lengths
        max_possible_pairs = max(m, n)
        return dp[m][n] / max_possible_pairs if max_possible_pairs > 0 else 0.0

    def verify_forward(
        self,
        correct_frame_id_sequence: List[str],
        correct_state_idx_sequence: List[int],
        input_state_idx_sequence: List[int],
        return_signatures: bool = False
    ) -> Tuple[Dict[str, Any], Union[Dict[str, Any], None]]:
        """
        Verify if the input sequence is correct for forward dynamics.
        
        Args:
            correct_frame_id_sequence: The correct (unshuffled) frame sequence [x, y, z]
            correct_state_idx_sequence: Ground truth answer (1-indexed)
            input_state_idx_sequence: Predicted answer (1-indexed)
            
        Returns:
            Dictionary containing evaluation results with raw scores
        """
        # Reconstruct the actual shuffled sequence from the correct sequence and correct indices
        # correct_frame_id_sequence is the correct sequence [x, y, z]
        # correct_state_idx_sequence tells us the mapping to reconstruct shuffled sequence
        
        # Reconstruct what the shuffled sequence should look like
        # If correct_state_idx_sequence is [2, 1], it means shuffled[2]=y, shuffled[1]=z
        # So shuffled sequence should be [x, z, y] to make [shuffled[0], shuffled[2], shuffled[1]] = [x, y, z]
        actual_shuffled_sequence = [None] * len(correct_frame_id_sequence)
        actual_shuffled_sequence[0] = correct_frame_id_sequence[0]  # Current state is always first
        
        # Reconstruct the shuffled positions
        for i, correct_idx in enumerate(correct_state_idx_sequence):
            if i + 1 < len(correct_frame_id_sequence):  # Skip if we're out of bounds
                actual_shuffled_sequence[correct_idx] = correct_frame_id_sequence[i + 1]
        
        # Fill any remaining None values (shouldn't happen in correct data)
        for i in range(len(actual_shuffled_sequence)):
            if actual_shuffled_sequence[i] is None:
                raise ValueError(f"Actual shuffled sequence is not fully reconstructed. Missing frame at index {i}")

        correct_sequence = correct_state_idx_sequence
        input_sequence = input_state_idx_sequence

        # Calculate exact match
        exact_match = correct_sequence == input_sequence
        equal_length = len(correct_sequence) == len(input_sequence)
        
        # Initialize all variables at method level to avoid scope issues
        semantic_match = False
        relative_correct_ratio = 0
        correct_signatures = []
        input_signatures = []
        correct_frame_id_sequence = []
        input_frame_id_sequence = []
        if equal_length:
            try:
                # Convert indices to frame sequences using the reconstructed shuffled sequence
                correct_frame_id_sequence = [actual_shuffled_sequence[idx] for idx in correct_sequence]
                input_frame_id_sequence = [actual_shuffled_sequence[idx] for idx in input_sequence]

                # append the current state to the frame id sequence at the beginning
                correct_frame_id_sequence.insert(0, actual_shuffled_sequence[0])
                input_frame_id_sequence.insert(0, actual_shuffled_sequence[0])

                # Get signatures for both sequences
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                input_signatures = self._translate_sequence_to_signatures(input_frame_id_sequence, partial_diff=False)

                # Check if all correct signatures are subsets of input signatures
                semantic_match = True
                for i in range(len(correct_signatures)):
                    # as long as the action sequence(ground truth) can describe the input sequence, it is correct
                    if not correct_signatures[i].issubset(input_signatures[i]):
                        semantic_match = False
                        break

                # Original approach for equal length sequences
                correct_pairs = 0
                for i in range(len(correct_signatures)):
                    if correct_signatures[i].issubset(input_signatures[i]):
                        correct_pairs += 1
                total_pairs = len(correct_sequence)
                relative_correct_ratio = correct_pairs / total_pairs if total_pairs > 0 else 0
            except Exception as e:
                semantic_match = False

        # Combined match result: True if either exact OR semantic match
        match = exact_match or semantic_match

        # calculate relative correct ratio for adjacent two states
        if not equal_length:
            # New approach for different length sequences using alignment
            try:
                # Convert indices to frame sequences using the reconstructed shuffled sequence
                correct_frame_id_sequence = [actual_shuffled_sequence[idx] for idx in correct_sequence]
                input_frame_id_sequence = [actual_shuffled_sequence[idx] for idx in input_sequence]
                correct_frame_id_sequence.insert(0, actual_shuffled_sequence[0])
                input_frame_id_sequence.insert(0, actual_shuffled_sequence[0])

                # Get signatures for both sequences
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                input_signatures = self._translate_sequence_to_signatures(input_frame_id_sequence, partial_diff=False)

                # Use alignment-based calculation
                relative_correct_ratio = self._calculate_pair_correct_ratio_with_alignment(
                    correct_signatures, input_signatures
                )
            except Exception as e:
                relative_correct_ratio = 0


        if match:
            relative_correct_ratio = 1

        results = {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'match': match,  # Combined match result
            'pair_correct_ratio': relative_correct_ratio
        }
        if return_signatures:
            correct_signatures = [list(signature) for signature in correct_signatures]
            if equal_length:
                input_visible_signatures = self._translate_sequence_to_signatures(input_frame_id_sequence, partial_diff=True)
                ## convert to serializable format
                input_visible_signatures = [list(signature) for signature in input_visible_signatures]
                input_signatures = [list(signature) for signature in input_signatures]
                sig_dict = {
                    'equal_length': equal_length,
                    'correct_signatures': correct_signatures,
                    'input_signatures': input_visible_signatures
                }
                return results, sig_dict
            else:
                sig_dict = {
                    'equal_length': equal_length,
                    'correct_signatures': correct_signatures,
                    'input_signatures': []
                }
                return results, sig_dict
        return results, None

    def verify_inverse(
        self,
        correct_frame_id_sequence: List[str],
        correct_action_idx_sequence: List[int],
        input_action_idx_sequence: List[int],
        return_signatures: bool = False
    ) -> Tuple[Dict[str, Any], Union[Dict[str, Any], None]]:
        """
        Verify inverse dynamics ordering
        In the future, this could implement different logic for inverse dynamics.
        """
        # Convert to 0-indexed (this is correct implementation, as following part are all working on the diff signatures, which is 1 less than correct frame id sequence)
        correct_sequence = [idx - 1 for idx in correct_action_idx_sequence]
        input_sequence = [idx - 1 for idx in input_action_idx_sequence]


        # Calculate exact match
        exact_match = correct_sequence == input_sequence
        equal_length = len(correct_sequence) == len(input_sequence)
        
        # Initialize all variables at method level to avoid scope issues
        semantic_match = False
        relative_correct_ratio = 0
        correct_signatures = []
        correct_full_signatures = []
        input_signatures = []
        shuffled_signatures = []
        if equal_length:
            try:
                # Get signatures ordered by the correct action sequence
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                correct_full_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=False)
                
                # Example: if signatures are [x, y, z] ordered by correct action sequence
                # and correct action idx sequence is [3, 1, 2] (1-indexed) -> [2, 0, 1] (0-indexed)
                # This means: action 3 comes first, then action 1, then action 2
                # So the shuffled signatures would be ordered as [z, x, y] (following indices [2, 0, 1])
                
                # Create a mapping from correct action indices to signature positions
                # correct_sequence contains the 0-indexed positions in the shuffled order
                shuffled_signatures = [None] * len(correct_signatures)
                for i, action_idx in enumerate(correct_sequence):
                    if 0 <= action_idx < len(correct_signatures):
                        shuffled_signatures[action_idx] = correct_signatures[i]
                
                # Remove any None entries (in case of invalid indices)
                shuffled_signatures = [sig for sig in shuffled_signatures if sig is not None]
                
                # Now get the input signatures based on the input sequence using correct_signatures
                # The input signatures are a rearrangement of correct_signatures according to input_sequence
                input_signatures = [None] * len(shuffled_signatures)
                for i, action_idx in enumerate(input_sequence):
                    if 0 <= action_idx < len(shuffled_signatures):
                        input_signatures[i] = shuffled_signatures[action_idx]
                
                # Remove any None entries (in case of invalid indices)
                input_signatures = [sig for sig in input_signatures if sig is not None]
                
                # Check if all input signatures are subsets of correct signatures
                semantic_match = True
                if len(correct_signatures) != len(input_signatures):
                    semantic_match = False

                else:
                    for i in range(len(input_signatures)): # logic is different from foward dynamics
                        # as long as the input action sequence can describe the states, it is correct
                        # here, we do not force the action to be the visible action
                        if not input_signatures[i].issubset(correct_full_signatures[i]):
                            semantic_match = False
                            break

                correct_pairs = 0
                for i in range(len(input_signatures)):
                    if input_signatures[i].issubset(correct_full_signatures[i]):
                        correct_pairs += 1
                total_pairs = len(correct_sequence)
                relative_correct_ratio = correct_pairs / total_pairs if total_pairs > 0 else 0

            except Exception as e:
                semantic_match = False
        
        # Combined match result: True if either exact OR semantic match
        match = exact_match or semantic_match

        # calculate relative correct ratio for adjacent two states
        if not equal_length:
            # New approach for different length sequences using alignment
            try:
                # Get signatures for both sequences
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                correct_full_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=False)
                
                # Create a mapping from correct action indices to signature positions
                shuffled_signatures = [None] * len(correct_signatures)
                for i, action_idx in enumerate(correct_sequence):
                    if 0 <= action_idx < len(correct_signatures):
                        shuffled_signatures[action_idx] = correct_signatures[i]
                
                # Remove any None entries (in case of invalid indices)
                shuffled_signatures = [sig for sig in shuffled_signatures if sig is not None]
                
                # Now get the input signatures based on the input sequence
                input_signatures = [None] * len(shuffled_signatures)
                for i, action_idx in enumerate(input_sequence):
                    if 0 <= action_idx < len(shuffled_signatures):
                        input_signatures[i] = shuffled_signatures[action_idx]
                
                # Remove any None entries (in case of invalid indices)
                input_signatures = [sig for sig in input_signatures if sig is not None]
                
                # Use alignment-based calculation
                relative_correct_ratio = self._calculate_pair_correct_ratio_with_alignment(
                    correct_signatures, input_signatures, correct_full_signatures
                )
            except Exception as e:
                relative_correct_ratio = 0

        if match:
            relative_correct_ratio = 1

        results = {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'match': match,  # Combined match result
            'pair_correct_ratio': relative_correct_ratio
        }

        if return_signatures:
            # convert to serializable format
            correct_signatures = [list(signature) for signature in correct_signatures]
            if equal_length:
                correct_full_signatures = [list(signature) for signature in correct_full_signatures]
                input_signatures = [list(signature) for signature in input_signatures]
                sig_dict = {
                    'equal_length': equal_length,
                    'correct_signatures': correct_signatures,
                    'input_signatures': input_signatures
                }
                return results, sig_dict
            else:
                sig_dict = {
                    'equal_length': equal_length,
                    'correct_signatures': correct_signatures,
                    'input_signatures': []
                }
                return results, sig_dict

        return results, None


class OrderingEvaluator:
    """
    This class is the wrapper class for batch evaluation of ordering tasks.
    It processes JSONL files containing multiple data points and provides
    comprehensive evaluation metrics and reports.
    """
    
    def __init__(
        self,
        input_root_dir: str,
        raw_data_dir: str
    ):
        """
        Initialize the OrderingEvaluator.
        
        Args:
            input_root_dir (str): Root directory containing segmented scene graphs
            raw_data_dir (str): Directory containing raw data and scene graphs
        """
        self.input_root_dir = Path(input_root_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.eval_results = {}  # Cache for evaluation results
        self._verifiers_cache = {}  # Cache for TaskWiseOrderingVerifier instances
        self.skipped_items = []  # Track skipped items with reasons
        self.wrong_case_signatures = {}  # Track wrong case signatures
        
    def _parse_answer_string(self, answer_str: Union[str, list], gt_length: int) -> Optional[List[int]]:
        """
        Extract a Python list from the answer string.
        
        Args:
            answer_str (str): Raw answer string from model output
            
        Returns:
            Optional[List[int]]: Extracted list or None if parsing fails
        """
        if isinstance(answer_str, list):
            return answer_str
        
        if not answer_str or not isinstance(answer_str, str):
            return None
            
        try:
            # Try to find list patterns like [1, 2, 3] or [1,2,3]
            list_pattern = r'\[[\d\s,]+\]'
            matches = re.findall(list_pattern, answer_str)
            
            if matches:
                # Take the last match and evaluate it
                list_str = matches[-1]
                parsed_list = ast.literal_eval(list_str)

                if len(parsed_list) > gt_length:
                    diff = len(parsed_list) - gt_length
                    parsed_list = parsed_list[diff:]
                
                # Ensure it's a list of integers
                if isinstance(parsed_list, list) and all(isinstance(x, int) for x in parsed_list):
                    # some may start from 0, we need to convert to 1-indexed
                    if min(parsed_list) == 0:
                        parsed_list = [x + 1 for x in parsed_list]
                    # some may start from 2 or larger, we need to convert to 1-indexed
                    if min(parsed_list) > 1:
                        delta = min(parsed_list) - 1
                        parsed_list = [x - delta for x in parsed_list]
                    return parsed_list
                    
        except (SyntaxError, ValueError, TypeError):
            pass
            
        # Alternative: try to extract numbers separated by commas/spaces
        try:
            # Look for sequences of numbers
            numbers = re.findall(r'\d+', answer_str)
            if numbers:
                return [int(num) for num in numbers]
        except (ValueError, TypeError):
            pass
            
        return None
        
    def _create_task_data(self, task_name: str) -> TaskData:
        """
        Create TaskData object for a specific task.
        
        Args:
            task_name (str): Name of the task
            
        Returns:
            TaskData: Configured TaskData object
            
        Raises:
            FileNotFoundError: If required files are not found
        """
        segmented_scene_graph_file = self.input_root_dir / task_name / "segmented_scene_graph_0.json"
        raw_data_task_dir = self.raw_data_dir / task_name
        raw_scene_graph_file = raw_data_task_dir / "scene_graph_0.json"

        if not segmented_scene_graph_file.exists():
            raise FileNotFoundError(f"Segmented scene graph file not found: {segmented_scene_graph_file}")
        if not raw_scene_graph_file.exists():
            raise FileNotFoundError(f"Raw scene graph file not found: {raw_scene_graph_file}")
        
        # Load scene graph data using SceneGraphReader
        segmented_scene_graph_reader = SceneGraphReader(str(segmented_scene_graph_file))
        key_frame_ids = segmented_scene_graph_reader.get_available_frame_ids()

        raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file))

        # Collect image paths for each frame and sensor
        image_root_path, image_paths = self._collect_image_paths(raw_data_task_dir)

        task_data = TaskData(
            task_name=task_name,
            scene_graph_reader=raw_scene_graph_reader,
            key_frame_ids=key_frame_ids,
            image_paths=image_paths,
            task_dir=str(raw_data_task_dir),
            image_root_path=image_root_path
        )

        return task_data
        
    def _collect_image_paths(self, task_dir: Path, key_frame_ids=None) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Collect image paths for all key frames and sensors.
        This method is extracted from TaskWiseOrderingVerifier for reuse.
        
        Args:
            task_dir (Path): Path to the task directory
            key_frame_ids (List[str]): List of key frame IDs
            
        Returns:
            Tuple[str, Dict[str, Dict[str, str]]]: Image root path and mapping from frame_id to {sensor_name: image_path}
        """
        image_paths = {}
        
        # Find all sensor directories
        sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]

        image_root_path = task_dir.parent

        if key_frame_ids is None:
            # Collect all available frame IDs from sensor directories
            all_frame_ids = set()
            for sensor_dir in sensor_dirs:
                # Get all PNG files in this sensor directory
                for image_file in sensor_dir.glob("*.png"):
                    # Extract frame ID from filename (remove .png extension)
                    frame_id = image_file.stem
                    # Convert to int and back to string to ensure consistent formatting
                    try:
                        frame_id_int = int(frame_id)
                        all_frame_ids.add(str(frame_id_int))
                    except ValueError:
                        # Skip files that don't have numeric names
                        continue
            
            # Convert to sorted list for consistent ordering
            key_frame_ids = sorted(all_frame_ids, key=int)
        
        for frame_id in key_frame_ids:
            image_paths[frame_id] = {}
            
            for sensor_dir in sensor_dirs:
                sensor_name = sensor_dir.name
                # Frame files are named with 5-digit zero-padding (e.g., 00051.png)
                image_file = sensor_dir / f"{int(frame_id):05d}.png"
                
                if image_file.exists():
                    image_paths[frame_id][sensor_name] = str(image_file)
        
        return image_root_path, image_paths
        
    def _gather_verifiers(self, jsonl_path: str) -> Dict[str, TaskWiseOrderingVerifier]:
        """
        Scan the JSONL file to collect all unique task names and create verifiers.
        
        Args:
            jsonl_path (str): Path to the JSONL file
            
        Returns:
            Dict[str, TaskWiseOrderingVerifier]: Dictionary mapping task names to verifiers
        """
        unique_task_names = set()
        
        # First pass: collect all unique task names
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data_point = json.loads(line.strip())
                    task_name = data_point.get('task_name')
                    if task_name:
                        unique_task_names.add(task_name)
                except json.JSONDecodeError:
                    continue
                    
        # Create verifiers for each unique task
        verifiers = {}
        for task_name in unique_task_names:
            if task_name not in self._verifiers_cache:
                try:
                    # Create TaskData first
                    task_data = self._create_task_data(task_name)
                    # Pass TaskData to TaskWiseOrderingVerifier
                    verifier = TaskWiseOrderingVerifier(task_data=task_data)
                    self._verifiers_cache[task_name] = verifier
                    print(f"Created verifier for task: {task_name}")
                except Exception as e:
                    print(f"Failed to create verifier for task {task_name}: {str(e)}")
                    continue
                    
            verifiers[task_name] = self._verifiers_cache[task_name]
            
        return verifiers
        
    def evaluate(self, jsonl_path: str, analyze_wrong_case: bool = False) -> None:
        """
        Main evaluation method that processes the entire JSONL file.
        
        Args:
            jsonl_path (str): Path to the JSONL evaluation file
            analyze_wrong_case (bool): Whether to analyze wrong case signatures
        """
        print(f"Starting evaluation of {jsonl_path}")
        
        # Gather all required verifiers
        verifiers = self._gather_verifiers(jsonl_path)
        
        if not verifiers:
            print("No valid verifiers found. Evaluation aborted.")
            return
            
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each data point in the JSONL file
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    data_id = data_point.get('id')
                    
                    if not data_id:
                        print(f"Line {line_num}: Missing 'id' field, skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': 'unknown',
                            'line_num': line_num,
                            'reason': 'missing_id_field',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                        continue
                        
                    # Check if already evaluated
                    if data_id in self.eval_results and 'eval_metrics' in self.eval_results[data_id]:
                        skipped_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'already_evaluated',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                        continue
                        
                    # Extract required fields
                    task_name = data_point.get('task_name')
                    task_type = data_point.get('type')
                    key_frame_ids = data_point.get('key_frame_ids', [])
                    gt_answer = data_point.get('gt_answer', [])
                    answer = data_point.get('answer', '')
                    
                    # Validate required fields
                    if not task_name or task_name not in verifiers:
                        print(f"Line {line_num}: Invalid or missing task_name '{task_name}', skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'invalid_or_missing_task_name',
                            'task_name': task_name or 'missing',
                            'task_type': task_type
                        })
                        continue
                        
                    if not gt_answer or not key_frame_ids:
                        print(f"Line {line_num}: Missing gt_answer or key_frame_ids, skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'missing_gt_answer_or_key_frames',
                            'task_name': task_name,
                            'task_type': task_type
                        })
                        continue
                        
                    # Parse the model's answer
                    parsed_answer = self._parse_answer_string(answer, len(gt_answer))
                    if parsed_answer is None:
                        print(f"Line {line_num}: Failed to parse answer '{answer}', skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'failed_to_parse_answer',
                            'task_name': task_name,
                            'task_type': task_type,
                            'raw_answer': answer
                        })
                        continue
                        
                    # Perform verification
                    verifier = verifiers[task_name]

                    if data_id == "sorting_vegetables_1753928918914038_forward_dynamics_ordering_3_steps_5e2a15bf":
                        pass
                    
                    # Determine which verification method to use based on task type
                    if task_type and 'inverse' in task_type.lower():
                        eval_result, sig_dict = verifier.verify_inverse(key_frame_ids, gt_answer, parsed_answer, return_signatures=analyze_wrong_case)
                    else:
                        eval_result, sig_dict = verifier.verify_forward(key_frame_ids, gt_answer, parsed_answer, return_signatures=analyze_wrong_case)

                    if analyze_wrong_case:
                        wrong_case_entry = {
                            'id': data_id,
                            'type': task_type,
                            'task_name': task_name,
                            'key_frame_ids': key_frame_ids,
                            'gt_answer': gt_answer,
                            'parsed_answer': parsed_answer,
                            'raw_answer': answer,
                            'eval_metrics': eval_result,
                            'equal_length': sig_dict.get('equal_length', False),
                            'correct_signatures': sig_dict.get('correct_signatures', None),
                            'input_signatures': sig_dict.get('input_signatures', None)
                        }
                        self.wrong_case_signatures[data_id] = wrong_case_entry
                    
                    # Store the complete result
                    result_entry = {
                        'id': data_id,
                        'type': task_type,
                        'task_name': task_name,
                        'key_frame_ids': key_frame_ids,
                        'gt_answer': gt_answer,
                        'parsed_answer': parsed_answer,
                        'raw_answer': answer,
                        'eval_metrics': eval_result
                    }
                    
                    self.eval_results[data_id] = result_entry
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count} data points...")
                        
                except json.JSONDecodeError:
                    print(f"Line {line_num}: Invalid JSON format, skipping")
                    error_count += 1
                    self.skipped_items.append({
                        'id': 'unknown',
                        'line_num': line_num,
                        'reason': 'invalid_json_format',
                        'task_name': 'unknown',
                        'task_type': 'unknown'
                    })
                    continue
                except Exception as e:
                    print(f"Line {line_num}: Error during evaluation: {str(e)}")
                    error_count += 1
                    # Try to extract some info from the data_point if it exists
                    try:
                        data_point = json.loads(line.strip())
                        self.skipped_items.append({
                            'id': data_point.get('id', 'unknown'),
                            'line_num': line_num,
                            'reason': f'evaluation_error: {str(e)}',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                    except:
                        self.skipped_items.append({
                            'id': 'unknown',
                            'line_num': line_num,
                            'reason': f'evaluation_error: {str(e)}',
                            'task_name': 'unknown',
                            'task_type': 'unknown'
                        })
                    continue
                    
        print(f"Evaluation completed!")
        print(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        
    def report_skipped_items(self) -> Dict[str, Any]:
        """
        Generate a report of all skipped items with detailed reasons.
        
        Returns:
            Dict[str, Any]: Dictionary containing skipped items and summary statistics
        """
        if not self.skipped_items:
            return {
                'total_skipped': 0,
                'skipped_by_reason': {},
                'skipped_items': []
            }
        
        # Group by reason
        skipped_by_reason = defaultdict(list)
        for item in self.skipped_items:
            reason = item.get('reason', 'unknown')
            skipped_by_reason[reason].append(item)
        
        # Create summary statistics
        reason_summary = {}
        for reason, items in skipped_by_reason.items():
            reason_summary[reason] = {
                'count': len(items),
                'percentage': len(items) / len(self.skipped_items) * 100
            }
        
        return {
            'total_skipped': len(self.skipped_items),
            'skipped_by_reason': reason_summary,
            'skipped_items': self.skipped_items
        }
        
    def print_skipped_items_summary(self) -> None:
        """
        Print a human-readable summary of skipped items.
        """
        skipped_report = self.report_skipped_items()
        
        if skipped_report['total_skipped'] == 0:
            print("\n📋 SKIPPED ITEMS SUMMARY: No items were skipped!")
            return
        
        print(f"\n📋 SKIPPED ITEMS SUMMARY:")
        print(f"   Total Skipped: {skipped_report['total_skipped']}")
        print(f"\n   Skipped by Reason:")
        
        for reason, stats in skipped_report['skipped_by_reason'].items():
            print(f"     • {reason}: {stats['count']} items ({stats['percentage']:.1f}%)")
        
        print(f"\n   Detailed Skipped Items:")
        for item in skipped_report['skipped_items']:
            id_str = item.get('id', 'unknown')
            line_num = item.get('line_num', 'unknown')
            reason = item.get('reason', 'unknown')
            task_name = item.get('task_name', 'unknown')
            task_type = item.get('task_type', 'unknown')
            
            print(f"     • ID: {id_str} (Line {line_num})")
            print(f"       Reason: {reason}")
            print(f"       Task: {task_name} ({task_type})")
            
            # Add raw answer for parsing failures
            if 'raw_answer' in item:
                raw_answer = item['raw_answer']
                # Truncate long answers
                if len(raw_answer) > 100:
                    raw_answer = raw_answer[:100] + "..."
                print(f"       Raw Answer: {raw_answer}")
            print()
        
    def report_by_task_type(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate comprehensive statistics report grouped by task type (question_type and step_num).
        
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Nested dictionary {question_type: {step_num: {metric: value}}}
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Group by question type and step number
        type_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for result in self.eval_results.values():
            task_type = result.get('type', '')
            eval_metrics = result.get('eval_metrics', {})
            
            if not task_type or not eval_metrics:
                continue
                
            # Parse task type (e.g., "forward_dynamics_10_steps")
            parts = task_type.split('_')
            if len(parts) >= 3:
                question_type = '_'.join(parts[:-2])  # e.g., "forward_dynamics"
                step_info = '_'.join(parts[-2:])       # e.g., "10_steps"
                
                # Extract step number
                step_num = ''.join(filter(str.isdigit, step_info))
                if not step_num:
                    step_num = 'unknown'
                    
                # Collect all metrics
                type_groups[question_type][step_num]['match'].append(eval_metrics.get('match', False))
                type_groups[question_type][step_num]['exact_match'].append(eval_metrics.get('exact_match', False))
                type_groups[question_type][step_num]['semantic_match'].append(eval_metrics.get('semantic_match', False))
                type_groups[question_type][step_num]['pair_correct_ratio'].append(eval_metrics.get('pair_correct_ratio', 0.0))
                
        # Calculate comprehensive statistics for each group
        report = {}
        for question_type, step_groups in type_groups.items():
            report[question_type] = {}
            for step_num, metrics_data in step_groups.items():
                stats = {}
                
                # Calculate accuracy metrics
                match_results = metrics_data['match']
                exact_match_results = metrics_data['exact_match']
                semantic_match_results = metrics_data['semantic_match']
                pair_correct_ratio_values = metrics_data['pair_correct_ratio']
                stats['count'] = len(match_results)
                stats['overall_accuracy'] = sum(match_results) / len(match_results) if match_results else 0.0
                stats['exact_match_accuracy'] = sum(exact_match_results) / len(exact_match_results) if exact_match_results else 0.0
                stats['semantic_match_accuracy'] = sum(semantic_match_results) / len(semantic_match_results) if semantic_match_results else 0.0
                stats['pair_correct_ratio'] = sum(pair_correct_ratio_values) / len(pair_correct_ratio_values) if pair_correct_ratio_values else 0.0
                # Calculate correlation statistics
                if pair_correct_ratio_values:
                    stats['avg_pair_correct_ratio'] = np.mean(pair_correct_ratio_values)
                    stats['std_pair_correct_ratio'] = np.std(pair_correct_ratio_values)
                    stats['min_pair_correct_ratio'] = np.min(pair_correct_ratio_values)
                    stats['max_pair_correct_ratio'] = np.max(pair_correct_ratio_values)
                
                report[question_type][step_num] = stats
                
        return report
        
    def report_by_task_name(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive statistics report grouped by task name.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary {task_name: {metric: value}}
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Group by task name
        task_groups = defaultdict(lambda: defaultdict(list))
        
        for result in self.eval_results.values():
            task_name = result.get('task_name')
            eval_metrics = result.get('eval_metrics', {})
            
            if task_name and eval_metrics:
                # Collect all metrics
                task_groups[task_name]['match'].append(eval_metrics.get('match', False))
                task_groups[task_name]['exact_match'].append(eval_metrics.get('exact_match', False))
                task_groups[task_name]['semantic_match'].append(eval_metrics.get('semantic_match', False))
                task_groups[task_name]['pair_correct_ratio'].append(eval_metrics.get('pair_correct_ratio', 0.0))
        # Calculate comprehensive statistics for each task
        report = {}
        for task_name, metrics_data in task_groups.items():
            stats = {}
            
            # Calculate accuracy metrics
            match_results = metrics_data['match']
            exact_match_results = metrics_data['exact_match']
            semantic_match_results = metrics_data['semantic_match']
            pair_correct_ratio_values = metrics_data['pair_correct_ratio']
            stats['count'] = len(match_results)
            stats['overall_accuracy'] = sum(match_results) / len(match_results) if match_results else 0.0
            stats['exact_match_accuracy'] = sum(exact_match_results) / len(exact_match_results) if exact_match_results else 0.0
            stats['semantic_match_accuracy'] = sum(semantic_match_results) / len(semantic_match_results) if semantic_match_results else 0.0
            stats['pair_correct_ratio'] = sum(pair_correct_ratio_values) / len(pair_correct_ratio_values) if pair_correct_ratio_values else 0.0
            # Calculate correlation statistics
            if pair_correct_ratio_values:
                stats['avg_pair_correct_ratio'] = np.mean(pair_correct_ratio_values)
                stats['std_pair_correct_ratio'] = np.std(pair_correct_ratio_values)
                stats['min_pair_correct_ratio'] = np.min(pair_correct_ratio_values)
                stats['max_pair_correct_ratio'] = np.max(pair_correct_ratio_values)
            
            report[task_name] = stats
            
        return report
    
    def report_by_ablation_task(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Generate comprehensive statistics report grouped by ablation settings, dynamics type, and step settings.
        Prints detailed accuracy and correlation values for each ablation-dynamics-step combination.
        
        Returns:
            Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]: Nested dictionary {ablation_setting: {dynamics_type: {step_num: {metric: value}}}}
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Group by ablation setting, dynamics type, and step number
        ablation_dynamics_step_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        for result in self.eval_results.values():
            data_id = result.get('id', '')
            task_type = result.get('type', '')
            eval_metrics = result.get('eval_metrics', {})
            
            if not data_id or not task_type or not eval_metrics:
                continue
                
            # Extract ablation setting from ID (last part after final underscore)
            # e.g., "canning_food_1751278778230696_forward_dynamics_ordering_3_steps_9be5cb1f_same_no_name"
            ablation_setting = 'unknown'
            if '_' in data_id:
                ablation_setting = '_'.join(data_id.split('_')[9:])
            
            # Extract dynamics type from task_type
            dynamics_type = 'unknown'
            if 'forward' in task_type.lower():
                dynamics_type = 'forward_dynamics'
            elif 'inverse' in task_type.lower():
                dynamics_type = 'inverse_dynamics'
                
            # Parse task type to extract step number (e.g., "forward_dynamics_ordering_3_steps")
            parts = task_type.split('_')
            step_num = 'unknown'
            if len(parts) >= 2:
                # Look for step information in the task type
                for part in parts:
                    if 'step' in part.lower():
                        # Extract number from step part
                        step_digits = ''.join(filter(str.isdigit, part))
                        if step_digits:
                            step_num = step_digits
                            break
                
                # If no explicit step info, try to extract any number from the task type
                if step_num == 'unknown':
                    for part in parts:
                        if part.isdigit():
                            step_num = part
                            break
                            
            # Collect all metrics
            ablation_dynamics_step_groups[ablation_setting][dynamics_type][step_num]['match'].append(eval_metrics.get('match', False))
            ablation_dynamics_step_groups[ablation_setting][dynamics_type][step_num]['exact_match'].append(eval_metrics.get('exact_match', False))
            ablation_dynamics_step_groups[ablation_setting][dynamics_type][step_num]['semantic_match'].append(eval_metrics.get('semantic_match', False))
            ablation_dynamics_step_groups[ablation_setting][dynamics_type][step_num]['pair_correct_ratio'].append(eval_metrics.get('pair_correct_ratio', 0.0))
                
        # Calculate comprehensive statistics and print detailed results
        report = {}
        
        print("\n" + "="*80)
        print("📊 ABLATION TASK REPORT - DETAILED BREAKDOWN")
        print("="*80)
        
        # Calculate normalization values across all steps and dynamics types
        all_step_nums = set()
        all_dynamics_types = set()
        for ablation_setting, dynamics_groups in ablation_dynamics_step_groups.items():
            for dynamics_type, step_groups in dynamics_groups.items():
                all_dynamics_types.add(dynamics_type)
                all_step_nums.update(step_groups.keys())
        
        # Sort step numbers and dynamics types for consistent ordering
        sorted_step_nums = sorted(all_step_nums, key=lambda x: int(x) if x.isdigit() else float('inf'))
        sorted_dynamics_types = sorted(all_dynamics_types)
        
        # Calculate normalization factors for each dynamics type and step combination
        step_dynamics_normalizations = {}
        for dynamics_type in sorted_dynamics_types:
            step_dynamics_normalizations[dynamics_type] = {}
            for step_num in sorted_step_nums:
                step_accuracies = []
                step_pair_correct_ratios = []
                for ablation_setting, dynamics_groups in ablation_dynamics_step_groups.items():
                    if dynamics_type in dynamics_groups and step_num in dynamics_groups[dynamics_type]:
                        metrics_data = dynamics_groups[dynamics_type][step_num]
                        if metrics_data['match']:
                            accuracy = sum(metrics_data['match']) / len(metrics_data['match'])
                            step_accuracies.append(accuracy)
                        if metrics_data['pair_correct_ratio']:
                            avg_pair_correct_ratio = np.mean(metrics_data['pair_correct_ratio'])
                            step_pair_correct_ratios.append(avg_pair_correct_ratio)
                
                step_dynamics_normalizations[dynamics_type][step_num] = {
                    'avg_accuracy': np.mean(step_accuracies) if step_accuracies else 0.0,
                    'std_accuracy': np.std(step_accuracies) if step_accuracies else 0.0,
                    'avg_pair_correct_ratio': np.mean(step_pair_correct_ratios) if step_pair_correct_ratios else 0.0,
                    'std_pair_correct_ratio': np.std(step_pair_correct_ratios) if step_pair_correct_ratios else 0.0,
                    'ablation_count': len(step_accuracies)
                }
        
        # Print normalization summary for each dynamics type
        # for dynamics_type in sorted_dynamics_types:
        #     print(f"\n📈 STEP NORMALIZATION SUMMARY - {dynamics_type.upper().replace('_', ' ')}:")
        #     print(f"{'Step':<8} {'Avg Acc':<10} {'Std Acc':<10} {'Avg Pair Correct Ratio':<12} {'Std Pair Correct Ratio':<12} {'Settings':<8}")
        #     print("-" * 70)
        #     for step_num in sorted_step_nums:
        #         if step_num in step_dynamics_normalizations[dynamics_type]:
        #             norm = step_dynamics_normalizations[dynamics_type][step_num]
        #             print(f"{step_num:<8} {norm['avg_accuracy']:<10.4f} {norm['std_accuracy']:<10.4f} "
        #                   f"{norm['avg_pair_correct_ratio']:<12.4f} {norm['std_pair_correct_ratio']:<12.4f} {norm['ablation_count']:<8}")
        
        # Process each ablation setting and print detailed results
        sorted_ablation_settings = sorted(ablation_dynamics_step_groups.keys())
        
        for ablation_setting in sorted_ablation_settings:
            print(f"\n🧪 ABLATION SETTING: {ablation_setting}")
            print("=" * 60)
            
            dynamics_groups = ablation_dynamics_step_groups[ablation_setting]
            ablation_report = {}
            
            # Process each dynamics type within this ablation setting
            for dynamics_type in sorted_dynamics_types:
                if dynamics_type not in dynamics_groups:
                    continue
                    
                print(f"\n🔄 {dynamics_type.upper().replace('_', ' ')}:")
                print("-" * 60)
                
                step_groups = dynamics_groups[dynamics_type]
                dynamics_report = {}
                
                # Sort steps for this dynamics type
                dynamics_step_nums = sorted(step_groups.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                
                # Print header
                print(f"{'Step':<8} {'Count':<8} {'Accuracy':<10} {'Exact':<8} {'Semantic':<10} {'Pair Correct Ratio':<12}")
                print("-" * 70)
                
                for step_num in dynamics_step_nums:
                    metrics_data = step_groups[step_num]
                    stats = {}
                    
                    # Calculate accuracy metrics
                    match_results = metrics_data['match']
                    exact_match_results = metrics_data['exact_match']
                    semantic_match_results = metrics_data['semantic_match']
                    pair_correct_ratio_values = metrics_data['pair_correct_ratio']
                    stats['count'] = len(match_results)
                    stats['overall_accuracy'] = sum(match_results) / len(match_results) if match_results else 0.0
                    stats['exact_match_accuracy'] = sum(exact_match_results) / len(exact_match_results) if exact_match_results else 0.0
                    stats['semantic_match_accuracy'] = sum(semantic_match_results) / len(semantic_match_results) if semantic_match_results else 0.0
                    stats['pair_correct_ratio'] = sum(pair_correct_ratio_values) / len(pair_correct_ratio_values) if pair_correct_ratio_values else 0.0
                    
                    # Calculate correlation statistics
                    if pair_correct_ratio_values:
                        stats['avg_pair_correct_ratio'] = np.mean(pair_correct_ratio_values)
                        stats['std_pair_correct_ratio'] = np.std(pair_correct_ratio_values)
                        stats['min_pair_correct_ratio'] = np.min(pair_correct_ratio_values)
                        stats['max_pair_correct_ratio'] = np.max(pair_correct_ratio_values)
                    else:
                        stats['avg_pair_correct_ratio'] = 0.0
                        stats['std_pair_correct_ratio'] = 0.0
                        stats['min_pair_correct_ratio'] = 0.0
                        stats['max_pair_correct_ratio'] = 0.0

                    
                    # Add normalization information
                    if dynamics_type in step_dynamics_normalizations and step_num in step_dynamics_normalizations[dynamics_type]:
                        norm = step_dynamics_normalizations[dynamics_type][step_num]
                        stats['normalized_accuracy'] = (stats['overall_accuracy'] - norm['avg_accuracy']) / norm['std_accuracy'] if norm['std_accuracy'] > 0 else 0.0
                        stats['normalized_pair_correct_ratio'] = (stats['avg_pair_correct_ratio'] - norm['avg_pair_correct_ratio']) / norm['std_pair_correct_ratio'] if norm['std_pair_correct_ratio'] > 0 else 0.0
                    else:
                        stats['normalized_accuracy'] = 0.0
                        stats['normalized_pair_correct_ratio'] = 0.0
                    
                    dynamics_report[step_num] = stats
                    
                    # Print row
                    print(f"{step_num:<8} {stats['count']:<8} {stats['overall_accuracy']:<10.4f} "
                          f"{stats['exact_match_accuracy']:<8.4f} {stats['semantic_match_accuracy']:<10.4f} "
                          f"{stats['avg_pair_correct_ratio']:<12.4f}")
                
                ablation_report[dynamics_type] = dynamics_report
            
            report[ablation_setting] = ablation_report
        
        print("\n" + "="*80)
        print("📋 REPORT COMPLETE")
        print("="*80)
                
        return report
        
    def report_overall_score(self) -> Dict[str, Any]:
        """
        Generate comprehensive overall performance report with detailed metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing comprehensive overall metrics
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Collect metrics by dynamics type
        forward_metrics = defaultdict(list)
        inverse_metrics = defaultdict(list)
        all_metrics = defaultdict(list)
        
        for result in self.eval_results.values():
            task_type = result.get('type', '').lower()
            eval_metrics = result.get('eval_metrics', {})
            
            if not eval_metrics:
                continue
            
            # Collect all metrics for overall statistics
            for metric_name in ['exact_match', 'semantic_match', 'match', 'pair_correct_ratio']:
                metric_value = eval_metrics.get(metric_name, 0.0 if metric_name in ['pair_correct_ratio'] else False)
                all_metrics[metric_name].append(metric_value)
                
                if 'forward' in task_type:
                    forward_metrics[metric_name].append(metric_value)
                elif 'inverse' in task_type:
                    inverse_metrics[metric_name].append(metric_value)
        
        # Build comprehensive report
        report = {}
        
        # Overall statistics
        if all_metrics['match']:
            report['overall'] = {
                'count': len(all_metrics['match']),
                'overall_accuracy': sum(all_metrics['match']) / len(all_metrics['match']),
                'exact_match_accuracy': sum(all_metrics['exact_match']) / len(all_metrics['exact_match']),
                'semantic_match_accuracy': sum(all_metrics['semantic_match']) / len(all_metrics['semantic_match']),
                'avg_pair_correct_ratio': np.mean(all_metrics['pair_correct_ratio']),
                'std_pair_correct_ratio': np.std(all_metrics['pair_correct_ratio']),
                'min_pair_correct_ratio': np.min(all_metrics['pair_correct_ratio']),
                'max_pair_correct_ratio': np.max(all_metrics['pair_correct_ratio'])
            }
        
        # Forward dynamics statistics
        if forward_metrics['match']:
            report['forward_dynamics'] = {
                'count': len(forward_metrics['match']),
                'overall_accuracy': sum(forward_metrics['match']) / len(forward_metrics['match']),
                'exact_match_accuracy': sum(forward_metrics['exact_match']) / len(forward_metrics['exact_match']),
                'semantic_match_accuracy': sum(forward_metrics['semantic_match']) / len(forward_metrics['semantic_match']),
                'avg_pair_correct_ratio': np.mean(forward_metrics['pair_correct_ratio']),
                'std_pair_correct_ratio': np.std(forward_metrics['pair_correct_ratio']),
                'min_pair_correct_ratio': np.min(forward_metrics['pair_correct_ratio']),
                'max_pair_correct_ratio': np.max(forward_metrics['pair_correct_ratio'])
            }
        
        # Inverse dynamics statistics
        if inverse_metrics['match']:
            report['inverse_dynamics'] = {
                'count': len(inverse_metrics['match']),
                'overall_accuracy': sum(inverse_metrics['match']) / len(inverse_metrics['match']),
                'exact_match_accuracy': sum(inverse_metrics['exact_match']) / len(inverse_metrics['exact_match']),
                'semantic_match_accuracy': sum(inverse_metrics['semantic_match']) / len(inverse_metrics['semantic_match']),
                'avg_pair_correct_ratio': np.mean(inverse_metrics['pair_correct_ratio']),
                'std_pair_correct_ratio': np.std(inverse_metrics['pair_correct_ratio']),
                'min_pair_correct_ratio': np.min(inverse_metrics['pair_correct_ratio']),
                'max_pair_correct_ratio': np.max(inverse_metrics['pair_correct_ratio'])
            }
                
        return report
