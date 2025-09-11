"""
Inverse Dynamics Q&A Generator.

This module implements the InverseDynamicsGenerator class that generates 
"given images A and B, what happened?" type questions.
"""

import sys
import json
import os
import random
import copy
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path
from tqdm import tqdm
import numpy as np
# Add PIL imports for image processing
from PIL import Image, ImageDraw, ImageFont
import hashlib
# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

try:
    from EmbodiedVLM.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
    from EmbodiedVLM.utils.qa_prompt_template import inv_prompt, multi_inv_prompt, multi_inv_ordering_prompt
    from EmbodiedVLM.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Random seeds are now set per-task in the generate() methods for deterministic behavior

class InverseDynamicsGenerator(AbstractQAGenerator):
    """
    Generates inverse dynamics Q&A pairs.
    
    Inverse Dynamics: Given state A and state B, what happened?
    """
    
    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool=True):
        """
        Initialize the inverse dynamics generator.
        
        Args:
            qa_gen_logic: Optional logic specification (reserved for future use)
        """
        # Note: Seeds are set in the generate method for each task, not here
        
        self.translator = StateChangeTranslator(type="inverse_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.sensor_names = ["external_sensor1"]

    @property
    def qa_type(self) -> str:
        return "inverse_dynamics" if not self.qa_gen_logic else f"{self.qa_gen_logic}_inverse_dynamics"
    
    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Path to the visual prompt for this Q&A generator. Should be default to QA_images/[qa_type]/[images]

        Returns:
            str: Path to the visual prompt
        """
        # replace the image root path last folder with 'QA_images'
        return image_root_dir / 'BehaviorEQA' / self.qa_type
    
    def generate(self, task_data: TaskData) -> List[QAPair]:
        """
        Generate inverse dynamics Q&A pairs for a task.
        
        Args:
            task_data: Task data containing scene graphs and images
            
        Returns:
            List[QAPair]: Generated Q&A pairs
        """
        # Reset random seeds for deterministic behavior within this task
        random.seed(42)
        np.random.seed(42)
        
        qa_pairs = []
        key_frame_ids = task_data.key_frame_ids

        candidate_gt_frame_pairs = set()
        for i in range(len(key_frame_ids) - 1):
            for j in range(i + 1, len(key_frame_ids)):
                candidate_gt_frame_pairs.add((key_frame_ids[i], key_frame_ids[j]))

        # filter out pairs that:
        ## 1. have no visible state changes
        ## 2. have too much difference (> 5)
        ## 3. have multiple same category objects in the visible diff

        pairs_to_remove = []

        for frame_a_id, frame_b_id in list(candidate_gt_frame_pairs):
            visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
                pairs_to_remove.append((frame_a_id, frame_b_id))
                continue  # Skip the rest of the loop for this pair
            
            gt_desc = self.translator.translate_diff(visible_diff)
            total_diff = gt_desc.count(".") if gt_desc else 0

            if not (1 <= total_diff <= 10): # 5 to 10 diff are acceptable
                pairs_to_remove.append((frame_a_id, frame_b_id))
        
        # Remove all pairs that need to be removed
        for pair in pairs_to_remove:
            candidate_gt_frame_pairs.remove(pair)

        # now we have a list of candidate gt frame pairs.
        # we see if we can find enough distractor images for each candidate gt frame pair
        for frame_a_id, frame_b_id in tqdm(candidate_gt_frame_pairs, desc="Generating QA pairs"):
            try:
                visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names)
                images_a = task_data.image_paths.get(frame_a_id, {})
                images_b = task_data.image_paths.get(frame_b_id, {})

                if not images_a or not images_b:
                    continue

                # get the sensor name
                sensor_name = self.sensor_names[0] # default to "external_sensor1"

                if sensor_name not in images_a or sensor_name not in images_b:
                    continue

                image_a_path = images_a[sensor_name]
                image_b_path = images_b[sensor_name]

                # Generate the QA pair
                qa_pair = self._create_inverse_qa_pair(
                    task_data, frame_a_id, frame_b_id, image_a_path, image_b_path, visible_diff, candidate_gt_frame_pairs
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"Error generating inverse QA for frames {frame_a_id}-{frame_b_id}: {e}")
                continue

        return qa_pairs

    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """Helper function to add text label to an image and save it."""
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Setup font - make it larger for better visibility
            font_size = max(40, img.height // 10)  # Increased font size (was img.height // 20)
            try:
                # Try to use a standard font
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                try:
                    # Fallback to DejaVu font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except (OSError, IOError):
                    # Use default font if no system fonts available
                    font = ImageFont.load_default()
            
            # Text styling - changed to bright red text with white outline
            text_color = (255, 20, 20)   # Bright red text (was white)
            outline_color = (255, 255, 255)  # White outline (was black)
            outline_width = 3  # Slightly thicker outline
            
            # Position text at top-left corner with some padding
            x, y = 15, 15  # Slightly more padding
            
            # Draw text with outline for better visibility
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw the main text
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Save the processed image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # If processing fails, copy the original image
            import shutil
            shutil.copy2(image_path, output_path)

    def _create_visual_prompt_for_images(self, qa_id: str, cur_state_image: str, next_state_image: str, task_data: TaskData) -> Tuple[str, str]:
        """
        Create a visual prompt for the images.

        Args:
            qa_id: QA pair ID
            cur_state_image: Current state image path
            next_state_image: Next state image path
            task_data: Task data
            
        Returns:
            Tuple containing the new current state image path and list of new option image paths
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        task_name = task_data.task_name

        # Create the full output directory path
        output_dir = Path(new_base_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Process current state image
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))

        # Process next state image
        next_state_output_path = output_dir / f"{qa_id}_next_state.png"
        self._add_text_to_image(next_state_image, "Next State", str(next_state_output_path))
        
        return str(cur_state_output_path), str(next_state_output_path)
    
    def _create_inverse_qa_pair(self, task_data: TaskData, frame_a_id: str, frame_b_id: str,
                               image_a_path: str, image_b_path: str, 
                               ground_truth_diff: Dict[str, Any],
                               candidate_frame_pairs: Set[Tuple[str, str]]) -> QAPair:
        """
        Create an inverse dynamics QA pair.
        
        Args:
            task_data: Task data
            frame_a_id: Starting frame ID
            frame_b_id: Ending frame ID  
            image_a_path: Path to image A
            image_b_path: Path to image B
            ground_truth_diff: The ground truth difference between frames
            
        Returns:
            QAPair: Generated QA pair
        """
        # Generate question
        question = inv_prompt
        
        # Generate correct answer using state change translator
        correct_answer = self.translator.translate_diff(ground_truth_diff)

        correct_answer_num = correct_answer.count(".")
        
        # Generate distractor options
        distractor_options = self._generate_distractor_options(
            task_data, frame_a_id, frame_b_id, ground_truth_diff, candidate_frame_pairs
        )

        if len(distractor_options) < 3:
            # print(f"Not enough distractor options for {frame_a_id}-{frame_b_id}")
            return None
        
        # Combine all options
        all_options = [correct_answer] + distractor_options
        random.shuffle(all_options)
        correct_option_index = all_options.index(correct_answer)

        # convert all_options to A, B, C, D
        all_options = [chr(i + 65) + ". " + option for i, option in enumerate(all_options)]
        correct_option_index = chr(correct_option_index + 65)
        
        # Create QA pair
        qa_id = f"{task_data.task_name}_{self.qa_type}_{frame_a_id}_{frame_b_id}"

        # Create visual prompt for the images
        if self.visual_prompt:
            image_a_path, image_b_path = self._create_visual_prompt_for_images(qa_id, image_a_path, image_b_path, task_data)
        
        gt_answer = {
            "type": self.qa_type,
            "options": all_options,
            "correct_option": correct_option_index,
        }

        question = question.format(STATE_CHANGES_CHOICES="\n".join(all_options))
        
        qa_pair = QAPair(
            id=qa_id,
            images=[image_a_path, image_b_path],
            meta_info=[correct_answer_num],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _negate_part_of_diff(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negate part of the diff.
        """
        '''
        diff = {
            'add': {
                'nodes': [{'name': 'node1', 'states': ['Contact']}, {'name': 'node2', 'states': ['Contact']}],
                'edges': [{'from': 'node1', 'to': 'node2', 'states': ['Contact']}]
            },
            'remove': {
                'nodes': [{'name': 'node3', 'states': ['...']}],
                'edges': [{'from': 'node1', 'to': 'node2', 'states': ['Contact']}]
            }
        }
        '''

        assert 'add' in diff and 'remove' in diff, f"Diff must contain both add and remove operations: {diff}"

        negated_diff = {
            'add': {
                'nodes': [],
                'edges': []
            },
            'remove': {
                'nodes': [],
                'edges': []
            }
        }

        negated = False

        for operation in ['add', 'remove']:
            the_other_operation = 'add' if operation == 'remove' else 'remove'
            if operation in diff:
                for node in diff[operation]['nodes']:
                    # randomly decide if we negate the node
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['nodes'].append(node)
                        negated = True
                    else:
                        negated_diff[operation]['nodes'].append(node)
                for edge in diff[operation]['edges']:
                    # randomly decide if we negate the edge
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['edges'].append(edge)
                        negated = True
                    else:
                        negated_diff[operation]['edges'].append(edge)

        if not negated:
            return {
                'add': diff['remove'],
                'remove': diff['add']
            }
        
        return negated_diff
    
    def _get_fake_state_centric_diff(self, raw_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a fake state-centric diff from a raw graph.
        """
        diff = {
            "add": {'nodes': [], 'edges': []},
            "remove": {'nodes': [], 'edges': []}
        }
        added = False

        for node in raw_graph['nodes']:
            for state in node['states']:
                diff['add']['nodes'].append({
                    "name": node['name'],
                    "states": [state]
                })
                added = True
        for edge in raw_graph['edges']:
            for state in edge['states']:
                diff['add']['edges'].append({
                    "from": edge['from'],
                    "to": edge['to'],
                    "states": [state]
                })
                added = True
        
        if not added:
            return None
    
        return diff
    
    def _generate_distractor_options(self, task_data: TaskData, correct_frame_a: str, 
                                   correct_frame_b: str, ground_truth_diff: Dict[str, Any],
                                   candidate_frame_pairs: Set[Tuple[str, str]]) -> List[str]:
        """
        Generate distractor options for the multiple choice question.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame of correct answer
            correct_frame_b: Ending frame of correct answer
            ground_truth_diff: Ground truth difference to avoid
            
        Returns:
            List[str]: List of distractor descriptions
        """
        distractors = []
        ground_truth_desc = self.translator.translate_diff(ground_truth_diff)
        ground_truth_diff_num = ground_truth_desc.count(".")

        # Strategy 1: real states in both frames, but not the state change (eg. the door keeps being open)
        # Get all unchanged states in both frames
        unchanged_states_scene_graph = task_data.scene_graph_reader.get_unchanged_states(correct_frame_a, correct_frame_b, self.sensor_names)

        fake_diff = self._get_fake_state_centric_diff(unchanged_states_scene_graph)

        if fake_diff:
            fake_desc = self.translator.translate_diff(fake_diff)
            fake_desc_num = fake_desc.count(".")
            if fake_desc_num > ground_truth_diff_num:
                fake_desc = fake_desc[:-1]
                fake_desc_parts = fake_desc.split(". ")
                # randomly pick ground_truth_diff_num parts
                fake_desc_parts = random.sample(fake_desc_parts, ground_truth_diff_num)
                fake_desc = ". ".join(fake_desc_parts) + "."
            if fake_desc and fake_desc not in distractors:
                distractors.append(fake_desc)

        # Strategy 2: negate part of the ground truth answer
        negated_diff = self._negate_part_of_diff(ground_truth_diff)
        negated_desc = self.translator.translate_diff(negated_diff)
        if negated_desc and negated_desc not in distractors:
            distractors.append(negated_desc)
        
        # Strategy 3: Use diffs from other frame pairs
        my_candidate_frame_pairs = list(candidate_frame_pairs)
        random.shuffle(my_candidate_frame_pairs)
        while len(distractors) < 3 and len(my_candidate_frame_pairs) > 0:
            selected_frame_pair = my_candidate_frame_pairs.pop()
            frame_c_id, frame_d_id = selected_frame_pair
            if frame_c_id == correct_frame_a and frame_d_id == correct_frame_b:
                continue

            distractor_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_c_id, frame_d_id, self.sensor_names, partial_diff=True)
            if distractor_diff.get('type') == 'empty' or not self._has_meaningful_changes(distractor_diff):
                continue

            if task_data.scene_graph_reader.is_subset_diff(distractor_diff, ground_truth_diff):
                continue
            distractor_desc = self.translator.translate_diff(distractor_diff)
            distractor_diff_num = distractor_desc.count(".")
            
            # control the diff number, must be similar to ground truth. standard deviation is 0.3
            if abs(distractor_diff_num - ground_truth_diff_num) > 0.3 * ground_truth_diff_num:
                continue
            
            if distractor_desc and distractor_desc not in distractors:
                distractors.append(distractor_desc)

        return distractors
    
    def _is_in_description(self, description: str, descriptions: List[str]) -> bool:
        """
        Check if a description is in a list of descriptions.
        """
        templates = [
            "now becomes",
            "becomes",
            "changes to be",
            "transitions to be",
            "is no longer",
            "stopped being"
        ]

        all_descriptions = []
        for template in templates:
            description = description.replace(template, "")

        for desc in descriptions:
            for template in templates:
                desc = desc.replace(template, "")
            all_descriptions.append(desc)

        return any(description in desc for desc in all_descriptions)
    
    def _has_meaningful_changes(self, diff: Dict[str, Any]) -> bool:
        """
        Check if a diff contains meaningful changes worth asking about.
        
        Args:
            diff: Scene graph difference
            
        Returns:
            bool: True if changes are meaningful
        """
        if diff.get('type') == 'empty':
            return False
        
        # Check for any substantial changes
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # Node changes are always meaningful
                if diff[operation].get('nodes'):
                    return True
                
                # Edge changes are meaningful if not just Contact states
                for edge in diff[operation].get('edges', []):
                    states = edge.get('states', [])
                    non_contact_states = [s for s in states if 'Contact' not in s]
                    if non_contact_states:
                        return True
        
        return False
    

class MultiStepInverseDynamicsGenerator(AbstractQAGenerator):
    """
    Generates multi-step inverse dynamics Q&A pairs.
    
    Multi-Step Inverse Dynamics: Given a sequence of states [S0, S1, ..., Sn], what sequence of actions happened?
    """

    def __init__(self, qa_gen_logic: str = "multi-choice", visual_prompt: bool = True, step_length: int = 5, option_num: int = 4):
        """
        Initialize the multi-step inverse dynamics generator.
        """
        # Note: Seeds are set in the generate method for each task, not here
        
        self.translator = StateChangeTranslator(type="multi_inverse_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 30, "Step length for inverse dynamics should be between 2 and 30."
        self.sensor_names = ["external_sensor1"]
        self.option_num = option_num
        assert self.option_num >= 4, f"Option number should be at least 4. Got {self.option_num} instead."

    @property
    def qa_type(self) -> str:
        if self.qa_gen_logic == "ordering":
            return f"inverse_dynamics_ordering_{self.step_length}_steps"
        elif self.qa_gen_logic == "multi-choice" or self.qa_gen_logic == None:
            return f"inverse_dynamics_option_{self.step_length}_steps_{self.option_num}_choices"
        else:
            raise ValueError(f"Invalid QA generation logic: {self.qa_gen_logic}")

    def visual_prompt_path(self, image_root_dir) -> str:
        return image_root_dir / 'QA' / 'images' / self.qa_type
        
    def _has_meaningful_changes(self, diff: Dict[str, Any]) -> bool:
        """
        Check if a diff contains meaningful changes worth asking about.
        
        Args:
            diff: Scene graph difference
            
        Returns:
            bool: True if changes are meaningful
        """
        if diff.get('type') == 'empty':
            return False
        
        # Check for any substantial changes
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # Node changes are always meaningful
                if diff[operation].get('nodes'):
                    return True
                
                # Edge changes are meaningful if not just Contact states
                for edge in diff[operation].get('edges', []):
                    states = edge.get('states', [])
                    non_contact_states = [s for s in states if 'Contact' not in s]
                    if non_contact_states:
                        return True
        
        return False
    
    def _is_valid_transition(self, frame_a_id: str, frame_b_id: str, task_data: TaskData) -> bool:
        """
        Check if a transition from frame_a_id to frame_b_id is valid.
        Note: The memoization is now handled by the graph building process.
        """
        visible_diff = task_data.scene_graph_reader.get_visible_full_diff(
            frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
        )
        if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
            return False
        
        gt_desc = self.translator.translate_diff(visible_diff)
        total_diff = gt_desc.count(".") if gt_desc else 0
        
        return 1 <= total_diff <= 5
    
    def _build_valid_transitions_graph(self, key_frame_ids: List[str], task_data: TaskData) -> Dict[str, List[str]]:
        """
        Pre-computes all valid transitions and builds a graph (adjacency list).
        This is an O(N^2) operation but is performed only once.
        """
        num_frames = len(key_frame_ids)
        graph = {frame_id: [] for frame_id in key_frame_ids}
        
        print("Phase 1: Building valid transitions graph...")

        for i in tqdm(range(num_frames), desc="Building Graph"):
            for j in range(i + 1, num_frames):
                frame_a = key_frame_ids[i]
                frame_b = key_frame_ids[j]
                if self._is_valid_transition(frame_a, frame_b, task_data):
                    graph[frame_a].append(frame_b)
        return graph

    def _count_paths_with_dp(self, graph: Dict[str, List[str]], key_frame_ids: List[str], frame_to_index: Dict[str, int]) -> np.ndarray:
        """
        Phase 1: Uses Dynamic Programming to count paths of varying lengths.
        Returns a dp_table where dp_table[k][i] is the number of valid paths
        of length (k+1) ending at frame i.
        """
        num_frames = len(key_frame_ids)
        # dp_table[k][i] stores the number of paths of length k ending at frame i.
        # Lengths are 1-based, so we use step_length as the size.
        dp_table = np.zeros((self.step_length, num_frames), dtype=np.int64)

        # Base case: All paths of length 1
        dp_table[0, :] = 1

        print("Phase 2: Counting valid paths with Dynamic Programming...")
        # Fill the DP table layer by layer
        for k in tqdm(range(1, self.step_length), desc="DP Path Counting"): # k is length - 1
            for i in range(num_frames):
                current_frame = key_frame_ids[i]
                # To find paths of length k+1 ending at i, we look for predecessors.
                # A predecessor is a frame j that has a valid transition to i.
                # It's more efficient to iterate backward from i-1 to find predecessors.
                # This part is tricky. A reverse graph would be faster.
                # For now, we iterate all frames and check if they are a predecessor.
                for j in range(i):
                    predecessor_frame = key_frame_ids[j]
                    if current_frame in graph[predecessor_frame]:
                        dp_table[k, i] += dp_table[k - 1, j]
        
        return dp_table

    def _sample_paths_randomly(
        self,
        num_to_sample: int,
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str]
    ) -> List[List[str]]:
        """
        Phase 3 (New): Samples a specified number of paths using weighted random backtracking.
        """
        sampled_sequences = []
        num_frames = len(key_frame_ids)
        final_k = self.step_length - 1 # Index for the final step in dp_table

        # The population for our first choice is all frames, weighted by the number of paths ending at them.
        end_node_population = list(range(num_frames))
        end_node_weights = dp_table[final_k, :]
        
        # Normalize weights to handle potential floating point issues, although not strictly necessary for random.choices
        total_weight = np.sum(end_node_weights)
        if total_weight == 0:
            return [] # No paths to sample from
        
        print(f"Phase 3: Sampling {num_to_sample} paths using Weighted Random Backtracking...")

        def _get_one_random_path(start_node_idx: int) -> List[str]:
            """Helper to reconstruct one random path starting from the end."""
            path_reversed = [key_frame_ids[start_node_idx]]
            current_idx = start_node_idx
            
            # Backtrack from step_length-1 down to 1
            for k in range(final_k, 0, -1):
                # Find all valid predecessors for the current node
                predecessors = []
                weights = []
                for prev_idx in range(current_idx):
                    prev_frame = key_frame_ids[prev_idx]
                    current_frame = key_frame_ids[current_idx]
                    # Check 1: Is there a valid edge?
                    # Check 2: Does the DP table show valid paths leading to this predecessor?
                    if current_frame in graph[prev_frame] and dp_table[k - 1, prev_idx] > 0:
                        predecessors.append(prev_idx)
                        weights.append(dp_table[k - 1, prev_idx])
                
                # If no predecessors found, something is wrong, but we handle it.
                if not predecessors:
                    break 
                
                # Make a weighted random choice for the next node in the path
                chosen_predecessor_idx = random.choices(predecessors, weights=weights, k=1)[0]
                path_reversed.append(key_frame_ids[chosen_predecessor_idx])
                current_idx = chosen_predecessor_idx
                
            return list(reversed(path_reversed))

        # --- Main Sampling Loop ---
        # Select starting points for reconstruction (i.e., end points of sequences)
        chosen_end_node_indices = random.choices(end_node_population, weights=end_node_weights, k=num_to_sample)
        
        for end_node_idx in tqdm(chosen_end_node_indices, desc="Sampling Paths"):
            path = _get_one_random_path(end_node_idx)
            if len(path) == self.step_length: # Ensure a full path was generated
                sampled_sequences.append(path)

        return sampled_sequences
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> List[str]:
        """
        Translates a sequence of frame IDs into a list of action descriptions.
        Each description corresponds to a transition between two consecutive frames.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            action_desc = self.translator.translate_diff(diff)
            if not action_desc:
                action_desc = "No meaningful change is observed."
            action_descriptions.append(action_desc)
        return action_descriptions
    
    def _translate_diff_sequence_to_actions(self, task_data: TaskData, diff_sequence: List[Dict[str, Any]]) -> List[str]:
        """
        Translates a sequence of diffs into a list of action descriptions.
        """
        action_descriptions = []
        for diff in diff_sequence:
            action_desc = self.translator.translate_diff(diff)
            if not action_desc:
                action_desc = "No meaningful change is observed."
            action_descriptions.append(action_desc)
        return action_descriptions
    
    def _validate_not_all_subsets(self, task_data: TaskData, grounded_seq: List[str], candidate_seq: List[str]) -> bool:
        """
        Validate that subset of candidate_seq exists in grounded_seq.
        """
        if len(candidate_seq) != len(grounded_seq):
            return False
        
        all_subsets = True

        for i in range(len(grounded_seq) - 1):
            # Get the visible diff for the grounded sequence
            frame_a_id = grounded_seq[i]
            frame_b_id = grounded_seq[i+1]
            grounded_visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            grounded_current_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_a_id)
            grounded_next_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_b_id)
            # Get the visible diff for the candidate sequence
            frame_a_id = candidate_seq[i]
            frame_b_id = candidate_seq[i+1]
            candidate_visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            candidate_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_b_id)

            if candidate_visible_diff.get('type') == 'empty':
                return False
            
            if task_data.scene_graph_reader.has_similar_edges(grounded_visible_diff, candidate_visible_diff, grounded_current_scene_graph, grounded_next_scene_graph, candidate_scene_graph):
                return False
            
            # check if grounded_visible_diff is a subset of candidate_visible_diff
            if not task_data.scene_graph_reader.is_diff_subset_scene(grounded_visible_diff, candidate_scene_graph) and not task_data.scene_graph_reader.is_diff_subset_scene(candidate_visible_diff, grounded_next_scene_graph):
                all_subsets = False
        
        return not all_subsets

    def _generate_distractor_action_sequences_old(
        self,
        correct_action_sequence: List[str],
        correct_frame_sequence: List[str],
        all_valid_sequences: List[List[str]],
        task_data: TaskData
    ) -> List[List[str]]:
        """
        Generates 3 distractor action sequences based on the defined heuristics.
        1. action itself is correct, but the relationship between actions are incorrect
        2. action itself is incorrect
        3. both action and relationship between actions are correct, but does not describe the transition in the question
        """
        distractors = []
        
        # Heuristic 1: Shuffle the Steps (Temporal Scrambling) - 1/3 of distractors
        if len(correct_action_sequence) > 1:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Create a copy and swap only a pair of actions
                shuffled_sequence = list(correct_action_sequence)
                
                if len(shuffled_sequence) >= 2:
                    # Pick two different random indices to swap
                    idx1, idx2 = random.sample(range(len(shuffled_sequence)), 2)
                    shuffled_sequence[idx1], shuffled_sequence[idx2] = shuffled_sequence[idx2], shuffled_sequence[idx1]
                else:
                    # If only one action, can't swap, skip this attempt
                    continue
                
                # Check if this distractor is unique
                if shuffled_sequence not in distractors and shuffled_sequence != correct_action_sequence:
                    distractors.append(shuffled_sequence)
                    cur_num += 1

        # Heuristic 2: Single-Step Negation - 1/3 of distractors
        if len(distractors) < self.option_num - 1 and len(correct_action_sequence) > 0:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random step to negate
                step_to_negate = random.randint(0, len(correct_action_sequence) - 1)
                frame_a_id = correct_frame_sequence[step_to_negate]
                frame_b_id = correct_frame_sequence[step_to_negate + 1]
                
                try:
                    original_diff = task_data.scene_graph_reader.get_visible_full_diff(
                        frame_a_id, frame_b_id, self.sensor_names
                    )
                    
                    # Use the negation logic from the single-step generator
                    negated_diff = self._negate_part_of_diff(original_diff)
                    negated_desc = self.translator.translate_diff(negated_diff)
                    
                    if negated_desc and negated_desc != correct_action_sequence[step_to_negate]:
                        # Create a distractor sequence with one negated step
                        distractor_seq = list(correct_action_sequence)
                        distractor_seq[step_to_negate] = negated_desc
                        
                        # Check if this distractor is unique
                        if distractor_seq not in distractors and distractor_seq != correct_action_sequence:
                            distractors.append(distractor_seq)
                            cur_num += 1
                            
                except Exception as e:
                    # If negation fails for this step, try another
                    continue

        # Heuristic 3: Describing Unchanged States (Saliency Trap) - 1/3 of distractors
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random step to replace with unchanged state description
                step_to_replace = random.randint(0, len(correct_action_sequence) - 1)
                frame_a_id = correct_frame_sequence[step_to_replace]
                frame_b_id = correct_frame_sequence[step_to_replace + 1]
                
                try:
                    # Get unchanged states for this specific transition
                    unchanged_states = task_data.scene_graph_reader.get_unchanged_states(frame_a_id, frame_b_id, self.sensor_names)
                    
                    # Create fake diff using the unchanged states logic from InverseDynamicsGenerator
                    fake_diff = self._get_fake_state_centric_diff(unchanged_states)
                    if not fake_diff:
                        continue
                    
                    fake_desc = self.translator.translate_diff(fake_diff)
                    if fake_desc and fake_desc != correct_action_sequence[step_to_replace]:
                        # Control the length to match the original action
                        original_action_num = correct_action_sequence[step_to_replace].count(".")
                        fake_desc_num = fake_desc.count(".")
                        
                        if fake_desc_num > original_action_num and original_action_num > 0:
                            fake_desc = fake_desc[:-1]  # Remove trailing period
                            fake_desc_parts = fake_desc.split(". ")
                            # Randomly pick original_action_num parts
                            fake_desc_parts = random.sample(fake_desc_parts, min(original_action_num, len(fake_desc_parts)))
                            fake_desc = ". ".join(fake_desc_parts) + "."
                        
                        # Create distractor sequence with one replaced step
                        distractor_seq = list(correct_action_sequence)
                        distractor_seq[step_to_replace] = fake_desc
                        
                        # Check if this distractor is unique
                        if distractor_seq not in distractors and distractor_seq != correct_action_sequence:
                            distractors.append(distractor_seq)
                            cur_num += 1
                            
                except Exception as e:
                    # If unchanged state extraction fails for this step, try another
                    continue

        # Heuristic 4: Globally Incorrect Sequence (Fallback)
        candidate_pool = [seq for seq in all_valid_sequences if seq != correct_frame_sequence]
        random.shuffle(candidate_pool)
        while len(distractors) < self.option_num - 1 and candidate_pool:
            distractor_frame_seq = candidate_pool.pop()
            distractor_action_seq = self._translate_sequence_to_actions(task_data, distractor_frame_seq)
            if distractor_action_seq and distractor_action_seq not in distractors:
                distractors.append(distractor_action_seq)

        return distractors
    
    def _swap_one_state_between_two_diffs(
        self,
        diff_1: Dict[str, Any],
        diff_2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Swaps one state between two diffs.
        
        Args:
            diff_1: First diff dictionary
            diff_2: Second diff dictionary
            
        Returns:
            Tuple of modified diffs with one state swapped between them
        """
        def collect_states(diff):
            """Collect all states with their location info (type, operation, index, dict)"""
            states = []
            for operation in ['add', 'remove']:
                if operation in diff:
                    for i, node in enumerate(diff[operation].get('nodes', [])):
                        states.append(('node', operation, i, node))
                    for i, edge in enumerate(diff[operation].get('edges', [])):
                        states.append(('edge', operation, i, edge))
            return states
        
        states_1 = collect_states(diff_1)
        states_2 = collect_states(diff_2)
        
        if not states_1 or not states_2:
            return None, None
        
        # Randomly select different states for swapping, try at most 100 times
        for _ in range(100):
            state_1 = random.choice(states_1)
            state_2 = random.choice(states_2)
            
            # Ensure the selected states are different
            if state_1[3] != state_2[3]:  # Different content
                break
        else:
            return None, None  # Can't find different states
        
        # Create new diffs and perform the swap
        new_diff_1 = copy.deepcopy(diff_1)
        new_diff_2 = copy.deepcopy(diff_2)
        
        # Unpack state information
        type_1, op_1, idx_1, dict_1 = state_1
        type_2, op_2, idx_2, dict_2 = state_2
        
        # Remove states from their original locations
        if type_1 == 'node':
            new_diff_1[op_1]['nodes'].pop(idx_1)
        else:
            new_diff_1[op_1]['edges'].pop(idx_1)
            
        if type_2 == 'node':
            new_diff_2[op_2]['nodes'].pop(idx_2)
        else:
            new_diff_2[op_2]['edges'].pop(idx_2)
        
        # Place states in correct locations based on their type
        # Place dict_2 (from diff_2) into diff_1 in the correct type list
        if type_2 == 'node':
            new_diff_1[op_1]['nodes'].append(dict_2)
        else:
            new_diff_1[op_1]['edges'].append(dict_2)
            
        # Place dict_1 (from diff_1) into diff_2 in the correct type list  
        if type_1 == 'node':
            new_diff_2[op_2]['nodes'].append(dict_1)
        else:
            new_diff_2[op_2]['edges'].append(dict_1)
        
        return new_diff_1, new_diff_2
    
    def _are_different_diffs(
        self,
        task_data: TaskData,
        diff_sequence_1: List[Dict[str, Any]],
        diff_sequence_2: List[Dict[str, Any]]
    ) -> bool:
        """
        Checks if the diffs in diff_sequence_1 are different from the diffs in diff_sequence_2.
        """
        for idx_1 in range(len(diff_sequence_1)):
            if task_data.scene_graph_reader.diff_signature(diff_sequence_1[idx_1]) != task_data.scene_graph_reader.diff_signature(diff_sequence_2[idx_1]):
                return True
        return False
    
    def _does_not_contain_diff_sequence(
        self,
        task_data: TaskData,
        diff_seq_to_check: List[Dict[str, Any]],
        diff_seq_sequence: List[List[Dict[str, Any]]]
    ) -> bool:
        """
        Checks if diff_seq_to_check is not a subset of any diff_seq in diff_seq_sequence.
        """
        if len(diff_seq_sequence) == 0:
            return True
        
        for diff_seq in diff_seq_sequence:
            if self._are_different_diffs(task_data, diff_seq_to_check, diff_seq):
                return True
        return False
                
    
    def _generate_distractor_action_sequences(
        self,
        correct_frame_sequence: List[str],
        all_valid_sequences: List[List[str]],
        task_data: TaskData
    ) -> List[List[str]]:
        """
        Generates distractor action sequences based on the defined heuristics.
        """
        distractors = []

        correct_diff_sequence = []

        for i in range(len(correct_frame_sequence) - 1):
            frame_a_id = correct_frame_sequence[i]
            frame_b_id = correct_frame_sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            correct_diff_sequence.append(diff)

        # Heuristic 3: Replace with one unchanged state
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) // 3 * 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10

            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1

                # Choose a random step to replace
                step_to_replace = random.randint(0, len(correct_diff_sequence) - 1)
                diff_to_replace = correct_diff_sequence[step_to_replace]
                frame_a_id = correct_frame_sequence[step_to_replace]
                frame_b_id = correct_frame_sequence[step_to_replace+1]

                unchanged_states = task_data.scene_graph_reader.get_unchanged_states(frame_a_id, frame_b_id, self.sensor_names)
                fake_diff = self._get_fake_state_centric_diff(unchanged_states)
                if not fake_diff:
                    continue

                new_diff, _ = self._swap_one_state_between_two_diffs(diff_to_replace, fake_diff)

                distracted_diff_sequence = copy.deepcopy(correct_diff_sequence)
                distracted_diff_sequence[step_to_replace] = new_diff

                if self._does_not_contain_diff_sequence(task_data, distracted_diff_sequence, distractors):
                    distractors.append(distracted_diff_sequence)
                    cur_num += 1
                else:
                    continue

        # Heuristic 1: Select part of state and shuffle it
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) // 3 
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10

            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1

                # Create a copy and swap only a pair of actions
                if len(correct_diff_sequence) >= 2:
                    # Pick two different random indices to swap
                    idx1, idx2 = random.sample(range(len(correct_diff_sequence)), 2)
                    diff_1 = correct_diff_sequence[idx1]
                    diff_2 = correct_diff_sequence[idx2]
                    
                    new_diff_1, new_diff_2 = self._swap_one_state_between_two_diffs(diff_1, diff_2)
                    if new_diff_1 and new_diff_2:
                        distracted_diff_sequence = copy.deepcopy(correct_diff_sequence)
                        distracted_diff_sequence[idx1] = new_diff_1
                        distracted_diff_sequence[idx2] = new_diff_2

                        if self._does_not_contain_diff_sequence(task_data, distracted_diff_sequence, distractors):
                            distractors.append(distracted_diff_sequence)
                            cur_num += 1
                        else:
                            continue
                else:
                    continue

        # Heuristic 2: Select part of state and negate it
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10

            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random step to negate
                step_to_negate = random.randint(0, len(correct_diff_sequence) - 1)
                diff_to_negate = correct_diff_sequence[step_to_negate]

                negated_diff = self._negate_part_of_diff(diff_to_negate)

                distracted_diff_sequence = copy.deepcopy(correct_diff_sequence)
                distracted_diff_sequence[step_to_negate] = negated_diff

                if self._does_not_contain_diff_sequence(task_data, distracted_diff_sequence, distractors):
                    distractors.append(distracted_diff_sequence)
                    cur_num += 1
                else:
                    continue
        

        # Translate current distractors to action sequences
        distractors = [self._translate_diff_sequence_to_actions(task_data, distractor) for distractor in distractors]

        # Heuristic 4: Globally Incorrect Sequence (Fallback)
        candidate_pool = [seq for seq in all_valid_sequences if seq != correct_frame_sequence]
        random.shuffle(candidate_pool)
        while len(distractors) < self.option_num - 1 and candidate_pool:
            distractor_frame_seq = candidate_pool.pop()
            distractor_action_seq = self._translate_sequence_to_actions(task_data, distractor_frame_seq)
            if distractor_action_seq and distractor_action_seq not in distractors:
                distractors.append(distractor_action_seq)
        return distractors[:self.option_num - 1]
    
    def _draw_text_on_image(self, image: Image.Image, text: str) -> Image.Image:
        """Helper function to draw a styled label onto a PIL Image object."""
        draw = ImageDraw.Draw(image)
        
        # Font setup
        font_size = max(30, image.height // 12)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Style setup
        text_color = (255, 20, 20)
        outline_color = (255, 255, 255)
        outline_width = 2
        x, y = 15, 15

        # Draw text with outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        draw.text((x, y), text, font=font, fill=text_color)
        return image

    def _create_filmstrip_image(
        self, 
        image_paths: List[str], 
        output_path: str,
        frame_labels: List[str]
    ) -> None:
        """
        Stitches a sequence of images into a single horizontal "filmstrip" image,
        with each frame individually labeled.
        """
        if len(image_paths) != len(frame_labels):
            raise ValueError("The number of image paths and frame labels must be equal.")

        try:
            labeled_images = []
            for i, path in enumerate(image_paths):
                img = Image.open(path)
                # Draw the specific label for this frame
                labeled_img = self._draw_text_on_image(img, frame_labels[i])
                labeled_images.append(labeled_img)

            images = labeled_images
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            
            filmstrip = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                filmstrip.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            
            filmstrip.save(output_path)
        except Exception as e:
            print(f"Error creating labeled filmstrip for {output_path}: {e}")

    def _create_visual_prompt_for_filmstrip(
        self,
        qa_id: str,
        sequence_image_paths: List[str],
        task_data: TaskData
    ) -> str:
        """
        Creates and saves a single, fully labeled filmstrip image for the QA input.
        """
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        output_dir = Path(new_base_dir) / task_data.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filmstrip_output_path = output_dir / f"{qa_id}_input_filmstrip.png"
        
        # --- Generate labels for each frame ---
        num_frames = len(sequence_image_paths)
        frame_labels = ["Current State"]
        for i in range(1, num_frames):
            frame_labels.append(f"Next State {i}")
        # --- End of new logic ---

        self._create_filmstrip_image(
            image_paths=sequence_image_paths,
            output_path=str(filmstrip_output_path),
            frame_labels=frame_labels
        )
        return str(filmstrip_output_path)


    def _create_multistep_inverse_qa_pair(
        self,
        task_data: TaskData,
        correct_frame_sequence: List[str],
        all_valid_sequences: List[List[str]]
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step inverse dynamics question.
        """
        # 1. Generate the correct sequence of actions
        correct_action_sequence = self._translate_sequence_to_actions(task_data, correct_frame_sequence)
        
        # 2. Generate distractor action sequences
        distractor_sequences = self._generate_distractor_action_sequences(
            correct_frame_sequence, all_valid_sequences, task_data
        )
        if len(distractor_sequences) < self.option_num - 1:
            return None

        # 3. Combine, shuffle, and format options
        all_options = [correct_action_sequence] + distractor_sequences
        random.shuffle(all_options)
        correct_option_index = all_options.index(correct_action_sequence)
        correct_option_letter = chr(correct_option_index + 65)

        # Format for the prompt (e.g., "A. 1. Do X. 2. Do Y.")
        formatted_options = []
        for i, option_seq in enumerate(all_options):
            option_letter = chr(i + 65)
            # Number each step in the action sequence
            numbered_actions = [f"[Action {j+1}] {action}" for j, action in enumerate(option_seq)]
            formatted_options.append(f"{option_letter}. {' '.join(numbered_actions)}")

        # 4. Create the visual prompt (input filmstrip)
        qa_id = f"{task_data.task_name}_{self.qa_type}_{'_'.join(correct_frame_sequence)}"
        sensor_name = self.sensor_names[0]
        image_paths = [task_data.image_paths[frame_id][sensor_name] for frame_id in correct_frame_sequence]
        
        final_input_image = image_paths # Default if not using visual prompts
        if self.visual_prompt:
            final_input_image = [self._create_visual_prompt_for_filmstrip(qa_id, image_paths, task_data)]

        # 5. Assemble the final QAPair
        gt_answer = {
            "type": self.qa_type,
            "options": formatted_options,
            "correct_option": correct_option_letter,
        }
        
        question = multi_inv_prompt.format(STATE_CHANGES_CHOICES="\n".join(formatted_options))
        
        qa_pair = QAPair(
            id=qa_id,
            images=final_input_image, # This is a list with one path to the filmstrip
            question=question,
            task_name=task_data.task_name,
            key_frame_ids=correct_frame_sequence,
            # meta_info=[self.option_num],
            gt_answer=gt_answer
        )
        return qa_pair

    # You will need to add _negate_part_of_diff from InverseDynamicsGenerator here as well
    def _negate_part_of_diff(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negate part of the diff by randomly selecting one state to negate.
        """
        assert 'add' in diff and 'remove' in diff, f"Diff must contain both add and remove operations: {diff}"
        
        # Collect all available states with their location info
        available_states = []
        for operation in ['add', 'remove']:
            if operation in diff:
                for i, node in enumerate(diff[operation].get('nodes', [])):
                    available_states.append(('node', operation, i))
                for i, edge in enumerate(diff[operation].get('edges', [])):
                    available_states.append(('edge', operation, i))
        
        if not available_states:
            return diff  # No states to negate, return original diff
        
        # Randomly select one state to negate
        state_type, operation, index = random.choice(available_states)
        the_other_operation = 'add' if operation == 'remove' else 'remove'
        
        # Create a deep copy of the original diff
        negated_diff = copy.deepcopy(diff)
        
        # Move the selected state from its current operation to the opposite operation
        if state_type == 'node':
            # Remove from current operation and add to opposite operation
            moved_node = negated_diff[operation]['nodes'].pop(index)
            negated_diff[the_other_operation]['nodes'].append(moved_node)
        else:  # edge
            # Remove from current operation and add to opposite operation
            moved_edge = negated_diff[operation]['edges'].pop(index)
            negated_diff[the_other_operation]['edges'].append(moved_edge)
        
        return negated_diff

    def _get_fake_state_centric_diff(self, raw_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a fake state-centric diff from a raw graph. (Copied from InverseDynamicsGenerator)
        """
        diff = {
            "add": {'nodes': [], 'edges': []},
            "remove": {'nodes': [], 'edges': []}
        }
        added = False

        for node in raw_graph['nodes']:
            for state in node['states']:
                diff['add']['nodes'].append({
                    "name": node['name'],
                    "states": [state]
                })
                added = True
        for edge in raw_graph['edges']:
            for state in edge['states']:
                diff['add']['edges'].append({
                    "from": edge['from'],
                    "to": edge['to'],
                    "states": [state]
                })
                added = True
        
        if not added:
            return None
    
        return diff

    def generate(self, task_data: TaskData, num_to_sample: int=30, max_qa_num: int=25) -> List[QAPair]:
        """
        Generates multi-step inverse dynamics QA pairs.
        """
        # Reset random seeds for deterministic behavior within this task
        random.seed(42)
        np.random.seed(42)
        
        mode = self.qa_gen_logic if self.qa_gen_logic else "multi-choice"
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        if len(key_frame_ids) < self.step_length:
            return []

        # Steps 1, 2, 3: Find all valid sequences using your existing DP/sampling logic
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        frame_to_index = {frame_id: i for i, frame_id in enumerate(key_frame_ids)}
        dp_table = self._count_paths_with_dp(graph, key_frame_ids, frame_to_index)
        total_paths = dp_table[self.step_length - 1].sum()
        if total_paths == 0:
            print("No valid sequences found.")
            return []
        
        print(f"\nFound a total of {total_paths} valid sequences of length {self.step_length}.")
        actual_num_to_sample = min(num_to_sample, int(total_paths))
        all_valid_sequences = self._sample_paths_randomly(
            actual_num_to_sample, graph, dp_table, key_frame_ids
        )
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")

        
        # >>> NEW PART: Generate QA pairs from sampled sequences <<<
        qa_pairs = []
        print(f"Phase 4: Generating QA pairs from {len(all_valid_sequences)} sequences...")
        if mode == "multi-choice":
            for seq in tqdm(all_valid_sequences, desc="Generating Q&A"):
                try:
                    qa_pair = self._create_multistep_inverse_qa_pair(
                        task_data, seq, all_valid_sequences
                    )
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    import traceback
                    print(f"Error generating QA for sequence {seq}: {e}")
                    traceback.print_exc()
                    continue
        elif mode == "ordering":
            for seq in tqdm(all_valid_sequences, desc="Generating Ordering QA Pairs"):
                try:
                    qa_pair = self._create_ordering_qa_pair(task_data, seq)
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    import traceback
                    print(f"Error generating QA for sequence {seq}: {e}")
                    traceback.print_exc()
                    continue
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        print(f"\nGenerated {len(qa_pairs)} multi-step inverse dynamics QA pairs.")
        if max_qa_num:
            print(f"Truncating to {max_qa_num} QA pairs.")
            # do random sampling
            qa_pairs = random.sample(qa_pairs, min(max_qa_num, len(qa_pairs)))
        return qa_pairs
    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """Helper function to add text label to an image and save it."""
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Setup font - make it larger for better visibility
            font_size = max(40, img.height // 10)  # Increased font size (was img.height // 20)
            try:
                # Try to use a standard font
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                try:
                    # Fallback to DejaVu font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except (OSError, IOError):
                    # Use default font if no system fonts available
                    font = ImageFont.load_default()
            
            # Text styling - changed to bright red text with white outline
            text_color = (255, 20, 20)   # Bright red text (was white)
            outline_color = (255, 255, 255)  # White outline (was black)
            outline_width = 3  # Slightly thicker outline
            
            # Position text at top-left corner with some padding
            x, y = 15, 15  # Slightly more padding
            
            # Draw text with outline for better visibility
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw the main text
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Save the processed image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # If processing fails, copy the original image
            import shutil
            shutil.copy2(image_path, output_path)

    def generate_qa_id_hash(self, task_name: str, qa_type: str, sequence: List[str]) -> str:
        """
        Generate a hash for a QA pair ID.
        """
        sequence_str = '_'.join(sequence)
        sequence_hash = hashlib.sha256(sequence_str.encode()).hexdigest()[:8]
        return f"{task_name}_{qa_type}_{sequence_hash}"
    
    def _create_ordering_qa_pair(
        self,
        task_data: TaskData,
        correct_sequence: List[str]
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step inverse dynamics ordering question.
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        qa_id = self.generate_qa_id_hash(task_data.task_name, self.qa_type, correct_sequence)
        sensor_name = self.sensor_names[0]
        image_paths = [task_data.image_paths[frame_id][sensor_name] for frame_id in correct_sequence]

        next_states = correct_sequence[1:]

        # final_input_image = image_paths # Default if not using visual prompts
        # if self.visual_prompt:
        #     final_input_image = [self._create_visual_prompt_for_filmstrip(qa_id, image_paths, task_data)]

        
        start_image_path = image_paths[0]
        next_state_image_paths = image_paths[1:]

        final_start_image = start_image_path
        final_next_state_images = []
        if self.visual_prompt:
            image_root_dir = task_data.image_root_path.parent
            new_base_dir = self.visual_prompt_path(image_root_dir)
            output_dir = Path(new_base_dir) / task_data.task_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
            self._add_text_to_image(start_image_path, "Current State", str(cur_state_output_path))
            final_start_image = str(cur_state_output_path)

            for i, frame_id in enumerate(next_states):
                next_state_image_path = task_data.image_paths[frame_id][sensor_name]
                label = f"Next State {i+1}"
                next_state_output_path = output_dir / f"{qa_id}_next_state_{i+1}.png"
                self._add_text_to_image(next_state_image_path, label, str(next_state_output_path))
                final_next_state_images.append(str(next_state_output_path))
        else:
            for frame_id in next_states:
                image_path = task_data.image_paths[frame_id][sensor_name]
                final_next_state_images.append(image_path)

        all_images = [final_start_image] + final_next_state_images
        all_images = [str(Path(image_path).relative_to(image_root_dir)) for image_path in all_images]
        
        correct_action_sequence = self._translate_sequence_to_actions(task_data, correct_sequence)
        correct_order = []

        shuffled_action_sequence = correct_action_sequence[:]
        random.shuffle(shuffled_action_sequence)

        # find the correct order of actions
        # Track which positions have been used to handle duplicate actions correctly
        used_positions = set()
        for original_action in correct_action_sequence:
            # Find the next unused position of this action in the shuffled sequence
            for i, shuffled_action in enumerate(shuffled_action_sequence):
                if shuffled_action == original_action and i not in used_positions:
                    shuffled_position = i + 1
                    correct_order.append(shuffled_position)
                    used_positions.add(i)
                    break
            else:
                # This should never happen if the sequences are properly constructed
                raise ValueError(f"Could not find unused position for action: {original_action}")

        # Attach [Action i] to each shuffled action
        numbered_shuffled_actions = [f"[Action {i+1}] {action}" for i, action in enumerate(shuffled_action_sequence)]

        numbered_action = '\n'.join(numbered_shuffled_actions)
        question = multi_inv_ordering_prompt.format(SHUFFLED_ACTIONS=numbered_action)

        # Validate that correct_order contains sequential numbers from 1 to N
        expected_sequence = list(range(1, len(correct_action_sequence) + 1))
        if sorted(correct_order) != expected_sequence:
            raise ValueError(f"Invalid correct_order sequence: {correct_order}. "
                           f"Expected sorted: {expected_sequence}, got sorted: {sorted(correct_order)}")

        gt_answer = {
            "type": self.qa_type,
            "options": [],
            "correct_option": correct_order,
        }

        qa_pair = QAPair(
            id=qa_id,
            images=all_images,
            task_name=task_data.task_name,
            key_frame_ids=correct_sequence,
            question=question,
            gt_answer=gt_answer
        )
        return qa_pair


class MultiStepInverseDynamicsAblationGenerator(AbstractQAGenerator):
    """
    For Ablation study QA generation. Always use ordering as task type.
    """
    def __init__(
        self,
        visual_prompt: bool=False,
        step_length: int=3,
    ):
        random.seed(42)
        np.random.seed(42)
        self.translator = StateChangeTranslator(
            type='multi_inverse_dynamics'
        )
        self.qa_gen_logic = 'ordering'
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 30, f"Step length should be between 2 and 30. Got {self.step_length} instead."
        self.sensor_names = ['external_sensor1']
    
    @property
    def qa_type(self) -> str:
        return f"inverse_dynamics_ordering_{self.step_length}_steps"
    
    def _has_meaningful_changes(self, diff: Dict[str, Any]) -> bool:
        """
        Check if a diff contains meaningful changes worth asking about.
        
        Args:
            diff: Scene graph difference
            
        Returns:
            bool: True if changes are meaningful
        """
        if diff.get('type') == 'empty':
            return False
        
        # Check for any substantial changes
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # Node changes are always meaningful
                if diff[operation].get('nodes'):
                    return True
                
                # Edge changes are meaningful if not just Contact states
                for edge in diff[operation].get('edges', []):
                    states = edge.get('states', [])
                    non_contact_states = [s for s in states if 'Contact' not in s]
                    if non_contact_states:
                        return True
        
        return False
    
    def _is_valid_transition(self, frame_a_id: str, frame_b_id: str, task_data: TaskData) -> bool:
        """
        Check if a transition from frame_a_id to frame_b_id is valid.
        Note: The memoization is now handled by the graph building process.
        """
        visible_diff = task_data.scene_graph_reader.get_visible_full_diff(
            frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
        )
        if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
            return False
        
        gt_desc = self.translator.translate_diff(visible_diff)
        total_diff = gt_desc.count(".") if gt_desc else 0
        
        return 1 <= total_diff <= 5
    
    def _build_valid_transitions_graph(self, key_frame_ids: List[str], task_data: TaskData) -> Dict[str, List[str]]:
        """
        Pre-computes all valid transitions and builds a graph (adjacency list).
        This is an O(N^2) operation but is performed only once.
        """
        num_frames = len(key_frame_ids)
        graph = {frame_id: [] for frame_id in key_frame_ids}
        
        print("Phase 1: Building valid transitions graph...")

        for i in tqdm(range(num_frames), desc="Building Graph"):
            for j in range(i + 1, num_frames):
                frame_a = key_frame_ids[i]
                frame_b = key_frame_ids[j]
                if self._is_valid_transition(frame_a, frame_b, task_data):
                    graph[frame_a].append(frame_b)
        return graph

    def _count_paths_with_dp(self, graph: Dict[str, List[str]], key_frame_ids: List[str], frame_to_index: Dict[str, int]) -> np.ndarray:
        """
        Phase 1: Uses Dynamic Programming to count paths of varying lengths.
        Returns a dp_table where dp_table[k][i] is the number of valid paths
        of length (k+1) ending at frame i.
        """
        num_frames = len(key_frame_ids)
        # dp_table[k][i] stores the number of paths of length k ending at frame i.
        # Lengths are 1-based, so we use step_length as the size.
        dp_table = np.zeros((self.step_length, num_frames), dtype=np.int64)

        # Base case: All paths of length 1
        dp_table[0, :] = 1

        print("Phase 2: Counting valid paths with Dynamic Programming...")
        # Fill the DP table layer by layer
        for k in tqdm(range(1, self.step_length), desc="DP Path Counting"): # k is length - 1
            for i in range(num_frames):
                current_frame = key_frame_ids[i]
                # To find paths of length k+1 ending at i, we look for predecessors.
                # A predecessor is a frame j that has a valid transition to i.
                # It's more efficient to iterate backward from i-1 to find predecessors.
                # This part is tricky. A reverse graph would be faster.
                # For now, we iterate all frames and check if they are a predecessor.
                for j in range(i):
                    predecessor_frame = key_frame_ids[j]
                    if current_frame in graph[predecessor_frame]:
                        dp_table[k, i] += dp_table[k - 1, j]
        
        return dp_table

    def _sample_paths_randomly(
        self,
        num_to_sample: int,
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str]
    ) -> List[List[str]]:
        """
        Phase 3 (New): Samples a specified number of paths using weighted random backtracking.
        """
        sampled_sequences = []
        num_frames = len(key_frame_ids)
        final_k = self.step_length - 1 # Index for the final step in dp_table

        # The population for our first choice is all frames, weighted by the number of paths ending at them.
        end_node_population = list(range(num_frames))
        end_node_weights = dp_table[final_k, :]
        
        # Normalize weights to handle potential floating point issues, although not strictly necessary for random.choices
        total_weight = np.sum(end_node_weights)
        if total_weight == 0:
            return [] # No paths to sample from
        
        print(f"Phase 3: Sampling {num_to_sample} paths using Weighted Random Backtracking...")

        def _get_one_random_path(start_node_idx: int) -> List[str]:
            """Helper to reconstruct one random path starting from the end."""
            path_reversed = [key_frame_ids[start_node_idx]]
            current_idx = start_node_idx
            
            # Backtrack from step_length-1 down to 1
            for k in range(final_k, 0, -1):
                # Find all valid predecessors for the current node
                predecessors = []
                weights = []
                for prev_idx in range(current_idx):
                    prev_frame = key_frame_ids[prev_idx]
                    current_frame = key_frame_ids[current_idx]
                    # Check 1: Is there a valid edge?
                    # Check 2: Does the DP table show valid paths leading to this predecessor?
                    if current_frame in graph[prev_frame] and dp_table[k - 1, prev_idx] > 0:
                        predecessors.append(prev_idx)
                        weights.append(dp_table[k - 1, prev_idx])
                
                # If no predecessors found, something is wrong, but we handle it.
                if not predecessors:
                    break 
                
                # Make a weighted random choice for the next node in the path
                chosen_predecessor_idx = random.choices(predecessors, weights=weights, k=1)[0]
                path_reversed.append(key_frame_ids[chosen_predecessor_idx])
                current_idx = chosen_predecessor_idx
                
            return list(reversed(path_reversed))

        # --- Main Sampling Loop ---
        # Select starting points for reconstruction (i.e., end points of sequences)
        chosen_end_node_indices = random.choices(end_node_population, weights=end_node_weights, k=num_to_sample)
        
        for end_node_idx in tqdm(chosen_end_node_indices, desc="Sampling Paths"):
            path = _get_one_random_path(end_node_idx)
            if len(path) == self.step_length: # Ensure a full path was generated
                sampled_sequences.append(path)

        return sampled_sequences

    def _translate_sequence_to_signatures(
        self,
        sequence: List[str],
        task_data: TaskData,
        partial_diff: bool = True
    ) -> List[Set[str]]:
        """
        Translate a sequence of frame IDs into a sequence of signatures.
        """
        signatures = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=partial_diff
            )
            signature = self.translator.translate_diff_into_signatures(diff)
            signatures.append(signature)
        return signatures

    def generate(
        self,
        task_data: TaskData,
    ):
        return None
    
    def generate_candidates(
        self, 
        task_data: TaskData,
        visual_prompt: bool=False,
        num_to_sample: int=12000
    ) -> List[QAPair]:
        """
        Generate multi-step forward dynamics QA pairs for a task.
        """
        qa_pairs = []
        # Reset random seeds for deterministic behavior within this task
        random.seed(42)
        np.random.seed(42)
        mode = self.qa_gen_logic if self.qa_gen_logic else "multi-choice"
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        num_frames = len(key_frame_ids)

        if num_frames < self.step_length:
            return []

        # Step 1: Build the transition graph (same as before)
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        frame_to_index = {frame_id: i for i, frame_id in enumerate(key_frame_ids)}
        
        # Step 2: Run the DP counting pass
        dp_table = self._count_paths_with_dp(graph, key_frame_ids, frame_to_index)
        
        total_paths = dp_table[self.step_length - 1].sum()
        print(f"\nDP table computed. Found a total of {total_paths} valid sequences.")

        if total_paths == 0:
            return []

        if total_paths < num_to_sample:
            raise ValueError(f"Not enough valid sequences found. Found {total_paths} valid sequences, but requested {num_to_sample}.")
        
        # Step 3: do none repeated sampling
        num_to_sample_max = min(total_paths, num_to_sample*10)
        all_valid_sequences = self._sample_paths_randomly(
            num_to_sample_max, graph, dp_table, key_frame_ids
        )
        
        # Convert list of lists to list of tuples so they can be hashed for deduplication
        all_valid_sequences = list(set(tuple(seq) for seq in all_valid_sequences))
        # Convert back to list of lists
        all_valid_sequences = [list(seq) for seq in all_valid_sequences]
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")

        if len(all_valid_sequences) < num_to_sample:
            raise ValueError(f"Not enough valid sequences found. Found {len(all_valid_sequences)} valid sequences, but requested {num_to_sample}.")
        
        all_valid_sequences = random.sample(all_valid_sequences, num_to_sample)

        qa_pairs = []
        # Step 4: creating QA and signatures
        for seq in tqdm(all_valid_sequences, desc="Generating QA piars and signatures"):
            try:
                qa_pair = self._create_multistep_qa_candidates(task_data, seq, visual_prompt)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Error generating QA for sequence {seq}: {e}")
                traceback.print_exc()
                continue
        
        return qa_pairs
    
    def get_valid_qa_pairs(
        self,
        jsonl_path: str,
        task_data: TaskData
    ) -> List[Dict]:
        """
        Get valid QA pairs from a JSONL file.
        """
        candidates_dict = {}
        validate_dict_list = []
        with open(jsonl_path, "r") as f:
            for line in f:
                candidate_dict = json.loads(line)
                step_string = f"{self.step_length}_steps"
                if step_string in candidate_dict['id']:
                    candidates_dict[candidate_dict["id"]] = candidate_dict
        
        for _, candidate_dict in tqdm(candidates_dict.items(), desc="Validating QA pairs"):
            key_frame_ids = candidate_dict['key_frame_ids']
            cur_signatures_list = candidate_dict['options']
            # tranform signatures to List[Set[str]], current is List[List[str]]
            cur_signatures = [set(signature) for signature in cur_signatures_list]

            # get current signatures
            gt_signatures = self._translate_sequence_to_signatures(key_frame_ids, task_data, partial_diff=True)

            # check if current signatures are subsets of gt signatures
            is_subset = True
            for i in range(len(cur_signatures)):
                if not cur_signatures[i].issubset(gt_signatures[i]):
                    is_subset = True
                    break
            
            if is_subset:
                validate_dict_list.append(candidate_dict)
                
        print(f"Found {len(validate_dict_list)} valid QA pairs.")
        return validate_dict_list
    
    def generate_qa_id_hash(self, task_name: str, qa_type: str, sequence: List[str]) -> str:
        """
        Generate a hash for a QA pair ID.
        """
        sequence_str = '_'.join(sequence)
        sequence_hash = hashlib.sha256(sequence_str.encode()).hexdigest()[:8]
        return f"{task_name}_{qa_type}_{sequence_hash}"

    def _create_multistep_qa_candidates(
        self,
        task_data: TaskData,
        input_sequence: List[str],
        visual_prompt: bool=False
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step forward dynamics question.
        """
        assert len(input_sequence) > 2, "Input sequence must be at least 3 frames long"
        assert len(input_sequence) == self.step_length, "Input sequence must be equal to the step length"

        qa_id = self.generate_qa_id_hash(task_data.task_name, self.qa_type, input_sequence)

        signatures = self._translate_sequence_to_signatures(input_sequence, task_data, partial_diff=True)

        # transform signatures to serializable format
        serialized_signatures = [list(signature) for signature in signatures]

        gt_answer = {
            "type": f"{self.qa_type}",
            "options": serialized_signatures,
            "correct_option": []
        }

        qa_pair = QAPair(
            id=qa_id,
            images=[],
            task_name=task_data.task_name,
            key_frame_ids=input_sequence,
            question="",
            gt_answer=gt_answer
        )

        return qa_pair
        
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> List[str]:
        """
        Translates a sequence of frame IDs into a list of action descriptions.
        Each description corresponds to a transition between two consecutive frames.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            action_desc = self.translator.translate_diff(diff)
            if not action_desc:
                action_desc = "No meaningful change is observed."
            action_descriptions.append(action_desc)
        return action_descriptions
    
    def generate_mother_QA_pairs(
        self,
        task_data: TaskData,
        file_path: str
    ) -> List[QAPair]:
        """
        Generate mother QA pairs from a JSONL file.
        """
        qa_dicts = []
        with open(file_path, "r") as f:
            for line in f:
                qa_dicts.append(json.loads(line))
        
        mother_qa_pairs = []
        for qa_dict in qa_dicts:
            seq = qa_dict['key_frame_ids']
            qa_pair = self._create_no_image_ordering_qa_pair(task_data, seq)
            if qa_pair:
                mother_qa_pairs.append(qa_pair)
        return mother_qa_pairs
    
    def generate_ablation_images(
        self,
        mother_qa_file: str,
        ablation_base_dir: str,
        output_base_dir: str
    ) -> None:
        """
        Generate images for all ablation settings based on the mother QA template.
        Creates a single combined JSONL file with separate image folders per ablation.
        
        Args:
            mother_qa_file: Path to the mother QA template JSONL file
            ablation_base_dir: Base directory containing all ablation folders
            output_base_dir: Base directory for QA output structure
        """
        import os
        import re
        from pathlib import Path
        from PIL import Image, ImageDraw, ImageFont
        import json
        from tqdm import tqdm
        
        # Read mother QA template
        mother_qa_pairs = []
        with open(mother_qa_file, "r") as f:
            for line in f:
                mother_qa_pairs.append(json.loads(line))
        
        print(f"Loaded {len(mother_qa_pairs)} mother QA pairs")
        
        # Find all ablation folders
        ablation_base_path = Path(ablation_base_dir)
        ablation_folders = [d for d in ablation_base_path.iterdir() if d.is_dir() and '[' in d.name and ']' in d.name]
        
        print(f"Found {len(ablation_folders)} ablation settings")
        
        # Collect all QA pairs across all ablations
        all_qa_pairs_with_images = []
        
        for ablation_folder in tqdm(ablation_folders, desc="Processing ablation settings"):
            # Extract ablation setting name from folder name
            match = re.search(r'\[([^\]]+)\]', ablation_folder.name)
            if not match:
                print(f"Warning: Could not extract ablation setting from folder name: {ablation_folder.name}")
                continue
                
            ablation_setting = match.group(1)
            print(f"\nProcessing ablation setting: {ablation_setting}")
            
            # Setup paths for this ablation
            sensor_dir = ablation_folder / "external_sensor1"
            if not sensor_dir.exists():
                print(f"Warning: Sensor directory not found for {ablation_setting}")
                continue
                
            # Create output structure for this ablation
            task_name = "canning_food_1751278778230696"  # Extract from folder name if needed
            
            for qa_data in tqdm(mother_qa_pairs, desc=f"Generating images for {ablation_setting}", leave=False):
                try:
                    # Create new QA ID with ablation setting appended
                    original_id = qa_data['id']
                    new_id = f"{original_id}_{ablation_setting}"
                    
                    # Get key frame IDs and determine step count
                    key_frame_ids = qa_data['key_frame_ids']
                    step_count = len(key_frame_ids)
                    qa_type = qa_data['type']
                    
                    # Setup output directory structure - qa_type folder contains ablation folders
                    images_output_dir = Path(output_base_dir) / "images" / qa_type / f"{ablation_setting}_{task_name}"
                    images_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate images with captions
                    final_images = []
                    
                    # 1. Current state image (first frame)
                    start_frame_id = key_frame_ids[0]
                    start_image_path = sensor_dir / f"{int(start_frame_id):05d}.png"
                    if not start_image_path.exists():
                        print(f"Warning: Start image not found: {start_image_path}")
                        continue
                        
                    cur_state_output_path = images_output_dir / f"{original_id}_cur_state.png"
                    self._add_text_to_image(str(start_image_path), "Current State", str(cur_state_output_path))
                    final_images.append(f"QA_ablation/images/{qa_type}/{ablation_setting}_{task_name}/{cur_state_output_path.name}")
                    
                    # 2. Next state images (remaining frames) - in correct order for inverse dynamics
                    next_frame_ids = key_frame_ids[1:]
                    
                    for i, frame_id in enumerate(next_frame_ids):
                        next_image_path = sensor_dir / f"{int(frame_id):05d}.png"
                        if not next_image_path.exists():
                            print(f"Warning: Next state image not found: {next_image_path}")
                            continue
                            
                        label = f"Next State {i + 1}"
                        next_state_output_path = images_output_dir / f"{original_id}_next_state_{i + 1}.png"
                        self._add_text_to_image(str(next_image_path), label, str(next_state_output_path))
                        final_images.append(f"QA_ablation/images/{qa_type}/{ablation_setting}_{task_name}/{next_state_output_path.name}")
                    
                    # Create updated QA pair
                    updated_qa = qa_data.copy()
                    updated_qa['id'] = new_id
                    updated_qa['images'] = final_images
                    all_qa_pairs_with_images.append(updated_qa)
                    
                except Exception as e:
                    print(f"Error processing QA {qa_data.get('id', 'unknown')}: {e}")
                    continue
        
        # Save all QA pairs to a single combined JSONL file
        qa_output_dir = Path(output_base_dir)
        qa_output_dir.mkdir(parents=True, exist_ok=True)
        qa_output_file = qa_output_dir / "inverse_behavior_eqa_ordering.jsonl"
        
        with open(qa_output_file, "w") as f:
            for qa_pair in all_qa_pairs_with_images:
                f.write(json.dumps(qa_pair) + "\n")
        
        print(f"\nSaved {len(all_qa_pairs_with_images)} total QA pairs from {len(ablation_folders)} ablation settings to {qa_output_file}")
    
    def _create_no_image_ordering_qa_pair(
        self, 
        task_data: TaskData, 
        correct_sequence: List[str]
    ) -> QAPair:
        """
        Create a QA pair for the ordering task without images.
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        # Get images for the sequence
        sensor_name = self.sensor_names[0]
        image_paths = [task_data.image_paths[frame_id][sensor_name] for frame_id in correct_sequence]

        # Generate QA pair ID
        step_num = len(correct_sequence)
        qa_type = f"inverse_dynamics_ordering_{step_num}_steps"
        qa_id = self.generate_qa_id_hash(task_data.task_name, qa_type, correct_sequence)
                
        # Get correct action sequence  
        correct_action_sequence = self._translate_sequence_to_actions(task_data, correct_sequence)
        correct_order = []

        # Shuffle the action sequence
        shuffled_action_sequence = correct_action_sequence[:]
        random.shuffle(shuffled_action_sequence)

        # Find the correct order of actions
        # Track which positions have been used to handle duplicate actions correctly
        used_positions = set()
        for original_action in correct_action_sequence:
            # Find the next unused position of this action in the shuffled sequence
            for i, shuffled_action in enumerate(shuffled_action_sequence):
                if shuffled_action == original_action and i not in used_positions:
                    shuffled_position = i + 1
                    correct_order.append(shuffled_position)
                    used_positions.add(i)
                    break
            else:
                # This should never happen if the sequences are properly constructed
                raise ValueError(f"Could not find unused position for action: {original_action}")

        # Attach [Action i] to each shuffled action
        numbered_shuffled_actions = [f"[Action {i+1}] {action}" for i, action in enumerate(shuffled_action_sequence)]

        numbered_action = '\n'.join(numbered_shuffled_actions)
        question = multi_inv_ordering_prompt.format(SHUFFLED_ACTIONS=numbered_action)

        # Validate that correct_order contains sequential numbers from 1 to N
        expected_sequence = list(range(1, len(correct_action_sequence) + 1))
        if sorted(correct_order) != expected_sequence:
            raise ValueError(f"Invalid correct_order sequence: {correct_order}. "
                           f"Expected sorted: {expected_sequence}, got sorted: {sorted(correct_order)}")

        # Create the ground truth answer
        gt_answer = {
            "type": qa_type,
            "options": [],
            'correct_option': correct_order
        }        
        qa_pair = QAPair(
            id=qa_id,
            images=[],
            task_name=task_data.task_name,
            key_frame_ids=correct_sequence,
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """Helper function to add text label to an image and save it."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            # Open the image
            img = Image.open(image_path)
            
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Setup font - make it larger for better visibility
            font_size = max(40, img.height // 10)  # Increased font size (was img.height // 20)
            try:
                # Try to use a standard font
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                try:
                    # Fallback to DejaVu font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except (OSError, IOError):
                    # Use default font if no system fonts available
                    font = ImageFont.load_default()
            
            # Text styling - changed to bright red text with white outline
            text_color = (255, 20, 20)   # Bright red text (was white)
            outline_color = (255, 255, 255)  # White outline (was black)
            outline_width = 3  # Slightly thicker outline
            
            # Position text at top-left corner with some padding
            x, y = 15, 15  # Slightly more padding
            
            # Draw text with outline for better visibility
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw the main text
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Save the processed image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # If processing fails, copy the original image
            import shutil
            shutil.copy2(image_path, output_path)

def collect_image_paths(task_dir: Path, key_frame_ids=None) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """
    Collect image paths for all key frames and sensors.
    
    Args:
        task_dir (Path): Path to the task directory
        key_frame_ids (List[str]): List of key frame IDs
        
    Returns:
        Dict[str, Dict[str, str]]: Mapping from frame_id to {sensor_name: image_path}
    """
    image_paths = {}
    
    # Find all sensor directories
    sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]

    image_root_path = task_dir.parent # file structure: image_root/task_name/sensor_name/frame_id.png

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

if __name__ == "__main__":
    task_name = "canning_food_1751278778230696"
    raw_data_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/raw/canning_food_1751278778230696"
    raw_data_dir = Path(raw_data_dir)
    raw_scene_graph_file = raw_data_dir / "aperture_30_scene_graph_0.json"
    raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file))

    segmented_scene_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/original_segmented/canning_food_1751278778230696"
    segmented_scene_dir = Path(segmented_scene_dir)
    segmented_scene_file = segmented_scene_dir / "segmented_scene_graph_0.json"
    segmented_scene_reader = SceneGraphReader(str(segmented_scene_file))
    key_frame_ids = segmented_scene_reader.get_available_frame_ids()

    image_root_path, image_paths = collect_image_paths(raw_data_dir)

    task_data = TaskData(
        task_name="canning_food_1751278778230696",
        scene_graph_reader=raw_scene_graph_reader,
        key_frame_ids=key_frame_ids,
        image_paths=image_paths,
        task_dir=str(raw_data_dir),
        image_root_path=image_root_path
    )

    file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_original_ordering_samples.jsonl"

    
    # ablation step 1 generation
    # step_3_generator = MultiStepInverseDynamicsAblationGenerator(step_length=3)
    # step_3_qa_pairs = step_3_generator.generate_candidates(task_data)

    # step_6_generator = MultiStepInverseDynamicsAblationGenerator(step_length=6)
    # step_6_qa_pairs = step_6_generator.generate_candidates(task_data)

    # step_9_generator = MultiStepInverseDynamicsAblationGenerator(step_length=9)
    # step_9_qa_pairs = step_9_generator.generate_candidates(task_data)

    # all_qa_pairs = step_3_qa_pairs + step_6_qa_pairs + step_9_qa_pairs

    # with open(file_path, "w") as f:
    #     for qa_pair in all_qa_pairs:
    #         f.write(json.dumps(qa_pair.to_dict()) + "\n")

    # ablation step 2: validation
    ## ablation step 2.1: validate aperture 30
    aperture_30_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_aperture_30_ordering_samples.jsonl"
    # aperture_30_step_3_generator = MultiStepInverseDynamicsAblationGenerator(step_length=3)
    # aperture_30_step_3_qa_dict_list = aperture_30_step_3_generator.get_valid_qa_pairs(file_path, task_data)
    # aperture_30_step_6_generator = MultiStepInverseDynamicsAblationGenerator(step_length=6)
    # aperture_30_step_6_qa_dict_list = aperture_30_step_6_generator.get_valid_qa_pairs(file_path, task_data)
    # aperture_30_step_9_generator = MultiStepInverseDynamicsAblationGenerator(step_length=9)
    # aperture_30_step_9_qa_dict_list = aperture_30_step_9_generator.get_valid_qa_pairs(file_path, task_data)

    # aperture_30_qa_dict_list = aperture_30_step_3_qa_dict_list + aperture_30_step_6_qa_dict_list + aperture_30_step_9_qa_dict_list

    # with open(aperture_30_file_path, "w") as f:
    #     for qa_dict in aperture_30_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")

    ## ablation step 2.2: validate high camera
    high_camera_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_high_camera_ordering_samples.jsonl"
    # high_camera_step_3_generator = MultiStepInverseDynamicsAblationGenerator(step_length=3)
    # high_camera_step_3_qa_dict_list = high_camera_step_3_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)
    # high_camera_step_6_generator = MultiStepInverseDynamicsAblationGenerator(step_length=6)
    # high_camera_step_6_qa_dict_list = high_camera_step_6_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)
    # high_camera_step_9_generator = MultiStepInverseDynamicsAblationGenerator(step_length=9)
    # high_camera_step_9_qa_dict_list = high_camera_step_9_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)

    # high_camera_qa_dict_list = high_camera_step_3_qa_dict_list + high_camera_step_6_qa_dict_list + high_camera_step_9_qa_dict_list

    # with open(high_camera_file_path, "w") as f:
    #     for qa_dict in high_camera_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")

    ## ablation step 2.3: validate low camera
    low_camera_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_low_camera_ordering_samples.jsonl"
    # low_camera_step_3_generator = MultiStepInverseDynamicsAblationGenerator(step_length=3)
    # low_camera_step_3_qa_dict_list = low_camera_step_3_generator.get_valid_qa_pairs(high_camera_file_path, task_data)
    # low_camera_step_6_generator = MultiStepInverseDynamicsAblationGenerator(step_length=6)
    # low_camera_step_6_qa_dict_list = low_camera_step_6_generator.get_valid_qa_pairs(high_camera_file_path, task_data)
    # low_camera_step_9_generator = MultiStepInverseDynamicsAblationGenerator(step_length=9)
    # low_camera_step_9_qa_dict_list = low_camera_step_9_generator.get_valid_qa_pairs(high_camera_file_path, task_data)

    # low_camera_qa_dict_list = low_camera_step_3_qa_dict_list + low_camera_step_6_qa_dict_list + low_camera_step_9_qa_dict_list

    # with open(low_camera_file_path, "w") as f:
    #     for qa_dict in low_camera_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")

    # for step 3, 6, 9, each randomly sample 50 dicts from the file
    inverse_sampled_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_ablation_ordering_samples.jsonl"
    # inverse_dict = {
    #     "3": [],
    #     "6": [],
    #     "9": []
    # }
    # sampled_list = []
    # ## 1. read the low camera file
    # with open(low_camera_file_path, "r") as f:
    #     for line in f:
    #         current_data = json.loads(line)
    #         id = current_data['id']
    #         if '3_steps' in id:
    #             inverse_dict['3'].append(current_data)
    #         elif '6_steps' in id:
    #             inverse_dict['6'].append(current_data)
    #         elif '9_steps' in id:
    #             inverse_dict['9'].append(current_data)
    # ## 2. sample 50 dicts from the foward_dict
    # random.seed(20250831)
    # for key in inverse_dict.keys():
    #     inverse_dict[key] = random.sample(inverse_dict[key], 50)
    #     sampled_list.extend(inverse_dict[key])
    # ## 3. write the sampled list to the file
    # with open(inverse_sampled_file_path, "w") as f:
    #     for qa_dict in sampled_list:
    #         f.write(json.dumps(qa_dict) + "\n")

# if __name__ == "__main__":
#     task_name = "canning_food_1751278778230696"
#     raw_data_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/raw/canning_food_1751278778230696"
#     raw_data_dir = Path(raw_data_dir)
#     raw_scene_graph_file = raw_data_dir / "raw_scene_graph_0.json"
#     raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file))

#     segmented_scene_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/original_segmented/canning_food_1751278778230696"
#     segmented_scene_dir = Path(segmented_scene_dir)
#     segmented_scene_file = segmented_scene_dir / "segmented_scene_graph_0.json"
#     segmented_scene_reader = SceneGraphReader(str(segmented_scene_file))
#     key_frame_ids = segmented_scene_reader.get_available_frame_ids()

#     image_root_path, image_paths = collect_image_paths(raw_data_dir)

#     task_data = TaskData(
#         task_name="canning_food_1751278778230696",
#         scene_graph_reader=raw_scene_graph_reader,
#         key_frame_ids=key_frame_ids,
#         image_paths=image_paths,
#         task_dir=str(raw_data_dir),
#         image_root_path=image_root_path
#     )

#     inverse_sampled_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_ablation_ordering_samples.jsonl"

    # generate mother QA pairs
    mother_qa_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/inverse_mother_ordering_samples.jsonl"
    mother_qa_generator = MultiStepInverseDynamicsAblationGenerator(step_length=3)
    mother_qa_pairs = mother_qa_generator.generate_mother_QA_pairs(task_data, inverse_sampled_file_path)
    with open(mother_qa_path, "w") as f:
        for qa_pair in mother_qa_pairs:
            f.write(json.dumps(qa_pair.to_dict()) + "\n")
