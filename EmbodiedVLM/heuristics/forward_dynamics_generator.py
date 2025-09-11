"""
Forward Dynamics Q&A Generator.

This module implements the ForwardDynamicsGenerator class that generates 
"given initial image and action description, what is the result?" type questions.
"""

import itertools
import sys
import os
import random
import copy
import numpy as np
import json
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
from tqdm import tqdm
import hashlib
# Add PIL imports for image processing
from PIL import Image, ImageDraw, ImageFont

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

try:
    from EmbodiedVLM.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from EmbodiedVLM.utils.qa_prompt_template import fwd_prompt, multi_fwd_prompt, multi_fwd_ordering_prompt, multi_inv_ordering_prompt
    from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
    from EmbodiedVLM.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Random seeds are now set per-task in the generate() methods for deterministic behavior


class ForwardDynamicsGenerator(AbstractQAGenerator):
    """
    Generates forward dynamics Q&A pairs.
    
    Forward Dynamics: Given state A and a description of change, what is the final state?
    """
    
    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool=True):
        """
        Initialize the forward dynamics generator.
        
        Args:
            qa_gen_logic: Optional logic specification (reserved for future use)
        """
        # Note: Seeds are set in the generate method for each task, not here
        
        self.translator = StateChangeTranslator(type="forward_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.sensor_names = ["external_sensor1"]

    @property
    def qa_type(self) -> str:
        return "forward_dynamics" if not self.qa_gen_logic else f"{self.qa_gen_logic}_forward_dynamics"
    
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
        Generate forward dynamics Q&A pairs for a task.
        
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
        
        # Get all candidate ground truth cur state and next state pairs (not confined to consecutive frames)

        candidate_gt_frame_pairs = set()
        for i in range(len(key_frame_ids) - 1):
            for j in range(i + 1, len(key_frame_ids)):
                candidate_gt_frame_pairs.add((key_frame_ids[i], key_frame_ids[j]))
        
        # filter out pairs that:
        ## 1. have no visible state changes
        ## 1.1 current visible state changes is quite strict
        ## 1.2 all objects in the state (unary and binary) must be visible in both frames
        ## 1.3 visible object threshold is 0.01%
        ## 2. have too much difference (> 8)
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
                visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
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
                qa_pair = self._create_forward_qa_pair(
                    task_data, frame_a_id, frame_b_id, image_a_path, image_b_path, visible_diff
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"Error generating forward QA for frames {frame_a_id}-{frame_b_id}: {e}")
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
    
    def _create_visual_prompt_for_images(self, qa_id: str, cur_state_image: str,all_image_options: List[str], task_data: TaskData) -> Tuple[str, List[str]]:
        """
        Create a visual prompt for the images.

        Args:
            qa_id: QA pair ID
            cur_state_image: Current state image path
            all_image_options: List of image paths (4 options)
            task_data: Task data
            
        Returns:
            Tuple containing the new current state image path and list of new option image paths
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        task_name = task_data.task_name

        assert len(all_image_options) == 4, f"There should be 4 image options. Got {len(all_image_options)} instead."
        
        # Create the full output directory path
        output_dir = Path(new_base_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Process current state image
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))
        
        # Process option images
        option_labels = ["Option A", "Option B", "Option C", "Option D"]
        option_output_paths = []
        
        for i, image_path in enumerate(all_image_options):
            if i >= 4:  # Ensure we only process 4 options
                break
            
            option_output_path = output_dir / f"{qa_id}_option_{chr(65 + i)}.png"
            self._add_text_to_image(image_path, option_labels[i], str(option_output_path))
            option_output_paths.append(str(option_output_path))
        
        # Ensure we have exactly 4 options (pad with last image if necessary)
        while len(option_output_paths) < 4:
            if all_image_options:
                last_image = all_image_options[-1]
                missing_option_idx = len(option_output_paths)
                option_output_path = output_dir / f"{qa_id}_option_{chr(65 + missing_option_idx)}.png"
                self._add_text_to_image(last_image, option_labels[missing_option_idx], str(option_output_path))
                option_output_paths.append(str(option_output_path))
            else:
                break
        
        return str(cur_state_output_path), option_output_paths
    
    def _create_forward_qa_pair(self, task_data: TaskData, frame_a_id: str, frame_b_id: str,
                               image_a_path: str, image_b_path: str, 
                               ground_truth_diff: Dict[str, Any]) -> QAPair:
        """
        Create a forward dynamics QA pair.
        
        Args:
            task_data: Task data
            frame_a_id: Starting frame ID
            frame_b_id: Ending frame ID  
            image_a_path: Path to initial image
            image_b_path: Path to result image (correct answer)
            ground_truth_diff: The ground truth difference between frames
            
        Returns:
            QAPair: Generated QA pair
        """
        # Generate action description from the diff
        action_description = self.translator.translate_diff(ground_truth_diff)

        description_num = action_description.count(".")

        # Capitalize the first letter of the action description
        # action_description = action_description.capitalize()
        
        # Generate question with action description
        question = fwd_prompt.format(STATE_CHANGES=action_description)
        
        # Generate distractor image options
        distractor_images = self._generate_distractor_images(
            task_data, frame_a_id, frame_b_id, ground_truth_diff
        )

        if len(distractor_images) < 3:
            # print(f"Not enough distractor images for {frame_a_id}-{frame_b_id}")
            return None
        
        # Combine all image options
        all_image_options = [image_b_path] + distractor_images
        random.shuffle(all_image_options)
        correct_option_index = all_image_options.index(image_b_path)

        # convert correct_option_index to A, B, C, D
        correct_option_index = chr(correct_option_index + 65)

        # Create QA pair
        qa_id = f"{task_data.task_name}_{self.qa_type}_{frame_a_id}_{frame_b_id}"
        
        # Create and save the visual prompt for the images
        if self.visual_prompt:
            cur_state_image, option_images = self._create_visual_prompt_for_images(qa_id, image_a_path, all_image_options, task_data)
            image_a_path = cur_state_image
            all_image_options = option_images

        gt_answer = {
            "type": self.qa_type,
            "options": all_image_options,
            "correct_option": correct_option_index,
        }
        
        qa_pair = QAPair(
            id=qa_id,
            images=[image_a_path],
            meta_info=[description_num],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _find_diced_and_half_objects(self, visible_objects: Set[str]) -> Set[str]:
        """
        Find objects that are diced or halved versions of other objects.
        
        This method identifies:
        1. Objects containing 'diced' or 'half' keywords
        2. For 'half' objects: extracts base object name and adds related objects
        3. For 'diced' objects: finds all objects that contain the base name
        
        Args:
            visible_objects: Set of visible object names
            
        Returns:
            Set of objects that are diced/half versions or their related objects
        """
        # Start with objects that explicitly contain 'diced' or 'half'
        diced_and_half_objects = {
            obj for obj in visible_objects 
            if 'diced' in obj or 'half' in obj
        }
        
        # Process each object to find related base objects
        for obj in visible_objects:
            if 'half' in obj:
                # Extract base object name by removing the first and last parts
                # e.g., "prefix_apple_half" -> "apple"
                base_obj = '_'.join(obj.split('_')[1:-1])
                if base_obj in visible_objects:
                    diced_and_half_objects.add(base_obj)
                    
            elif 'diced' in obj:
                # Remove 'diced__' prefix to get base name
                base_obj = obj.replace('diced__', '')
                # Find all objects that contain this base name
                related_objects = {
                    other_obj for other_obj in visible_objects 
                    if base_obj in other_obj
                }
                diced_and_half_objects.update(related_objects)
        
        return diced_and_half_objects
    
    def _is_overlapped_diced(self, gt_diced_and_half_objects: Set[str], current_diced_and_half_objects: Set[str]) -> bool:
        """
        Check if the diced and half objects in the ground truth are overlapped with the current scene graph.
        """
        # main idea is to find all (parent, child) (child, child) exist
        # here, it means the (complete, half), (complete, diced), (half, diced), (complete, complete), (half, half), (diced, diced) relations can be found for each object in gt and current
        # if a object cannot be found to form a relation above, then it is not overlapped, thus false
        
        def extract_base_object_name(obj_name: str) -> str:
            """Extract the base object name from diced/half object names."""
            if 'diced__' in obj_name:
                return obj_name.replace('diced__', '')
            elif 'half' in obj_name and '_' in obj_name:
                # For half objects like "prefix_apple_half", extract "apple"
                parts = obj_name.split('_')
                if len(parts) >= 3:
                    return '_'.join(parts[1:-1])
            return obj_name
        
        def get_object_variants(base_name: str, all_objects: Set[str]) -> Dict[str, Set[str]]:
            """Get all variants (complete, half, diced) of a base object."""
            variants = {
                'complete': set(),
                'half': set(), 
                'diced': set()
            }
            
            for obj in all_objects:
                obj_base = extract_base_object_name(obj)
                if obj_base == base_name or base_name in obj:
                    if 'diced' in obj:
                        variants['diced'].add(obj)
                    elif 'half' in obj:
                        variants['half'].add(obj)
                    else:
                        # Complete object (no diced/half modifiers)
                        variants['complete'].add(obj)
            
            return variants
        
        # Get all unique base object names from both sets
        all_base_names = set()
        for obj in gt_diced_and_half_objects | current_diced_and_half_objects:
            base_name = extract_base_object_name(obj)
            all_base_names.add(base_name)
        
        # Check for valid relationships for each base object
        for base_name in all_base_names:
            gt_variants = get_object_variants(base_name, gt_diced_and_half_objects)
            current_variants = get_object_variants(base_name, current_diced_and_half_objects)
            
            # Check if we can form valid relationships
            # Valid relationships: (complete, half), (complete, diced), (half, diced), 
            # (complete, complete), (half, half), (diced, diced)
            
            has_valid_relationship = False
            
            # Check all possible relationship types
            relationship_types = [
                ('complete', 'half'),
                ('complete', 'diced'), 
                ('half', 'diced'),
                ('complete', 'complete'),
                ('half', 'half'),
                ('diced', 'diced')
            ]
            
            for gt_type, current_type in relationship_types:
                if gt_variants[gt_type] and current_variants[current_type]:
                    has_valid_relationship = True
                    break
            
            # If no valid relationship found for this base object, return False
            if not has_valid_relationship:
                return False
        
        return True
    
    def _generate_distractor_images(
        self,
        task_data: TaskData,
        correct_frame_a: str,
        correct_frame_b: str,
        ground_truth_diff: Dict[str, Any]
    ) -> List[str]:
        """
        Generate distractor image options for the forward dynamics question.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame of correct answer
            correct_frame_b: Ending frame of correct answer (correct result image)
            ground_truth_diff: Ground truth difference to generate alternatives for
            
        Returns:
            List[str]: List of distractor image paths
        """
        distractors = []
        available_frame_ids = task_data.key_frame_ids

        VISUAL_SIMILAR_FRAME_DISTANCE = 40
        sensor_name = self.sensor_names[0]

        candidate_images = []
        candidate_frame_ids = []
        for frame_id in available_frame_ids:
            images = task_data.image_paths.get(frame_id, {})
            if sensor_name in images:
                current_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_id)
                visible_diff_1 = task_data.scene_graph_reader.get_visible_full_diff(correct_frame_a, frame_id, self.sensor_names)
                visible_diff_2 = task_data.scene_graph_reader.get_visible_full_diff(correct_frame_b, frame_id, self.sensor_names)

                if visible_diff_1.get('type') == 'empty' or visible_diff_2.get('type') == 'empty':
                    continue

                # check if most of the objects (more than 50%) is visible in the current scene graph
                all_current_visible_objects = task_data.scene_graph_reader.get_all_visible_objects_in_graph(self.sensor_names, current_scene_graph)
                gt_diff_visible_objects = task_data.scene_graph_reader.get_visible_objects_from_diff(correct_frame_a, correct_frame_b, self.sensor_names)

                # Handle diced and half objects with more flexible matching
                gt_diced_and_half_objects = self._find_diced_and_half_objects(gt_diff_visible_objects)
                current_diced_and_half_objects = self._find_diced_and_half_objects(all_current_visible_objects)

                is_overlapped_diced = self._is_overlapped_diced(gt_diced_and_half_objects, current_diced_and_half_objects)
                if not is_overlapped_diced:
                    continue

                other_gt_objects = gt_diff_visible_objects - gt_diced_and_half_objects
                other_current_objects = all_current_visible_objects - current_diced_and_half_objects

                shared_objects = other_gt_objects & other_current_objects

                # Check if there's sufficient overlap considering diced/half objects
                if len(shared_objects) < 0.5 * len(other_gt_objects):
                    continue

                # Below is too strict.
                # if not gt_diff_visible_objects.issubset(all_current_visible_objects):
                #     print("Because of visible objects")
                #     continue

                # check if the ground truth diff is the subset of current scene graph
                if task_data.scene_graph_reader.is_diff_subset_scene(ground_truth_diff, current_scene_graph):
                    continue
                
                ground_truth_current_scene_graph = task_data.scene_graph_reader.get_scene_graph(correct_frame_a)
                ground_truth_next_scene_graph = task_data.scene_graph_reader.get_scene_graph(correct_frame_b)

                # filter out if visible diff involves multiple same category objects
                # how to do this? well, just see if there is any similar edge is okay.
                if task_data.scene_graph_reader.has_similar_edges(ground_truth_diff, visible_diff_1, ground_truth_current_scene_graph, ground_truth_next_scene_graph, current_scene_graph):
                    continue

                candidate_images.append(images[sensor_name])
                candidate_frame_ids.append(frame_id)
        
        if len(candidate_frame_ids) > 0:
            pass
        
        # Strategy 1: Try to get the closest frame for frame_b but with less state difference. (High priority)
        ## convert candidate_frame_ids to int
        candidate_frame_ids = [int(frame_id) for frame_id in candidate_frame_ids]
        candidate_frame_ids.sort()
        ## find the frames between frame_a and frame_b
        frames_between = [frame_id for frame_id in candidate_frame_ids if int(correct_frame_a) < frame_id < int(correct_frame_b)]
        ## add frames that are closest to and larger than frame_b
        larger_frames = [frame_id for frame_id in candidate_frame_ids if frame_id > int(correct_frame_b)]
        frames_between = frames_between[-2:] # only keep the last 2 frames
        frames_between.extend(larger_frames[:2]) # add the first 2 frames that are larger than frame_b
        random.shuffle(frames_between)
        
        while len(distractors) < 3 and frames_between:
            frame_id = frames_between.pop()
            images = task_data.image_paths.get(str(frame_id), {})
            if sensor_name in images:
                distractors.append(images[sensor_name])

        # Strategy 2: Get closest visual similar frames for frame_a. Tipycally 20 frames away. Actually not a good strategy.
        if len(distractors) < 3:
            random_number = random.choice([-VISUAL_SIMILAR_FRAME_DISTANCE, VISUAL_SIMILAR_FRAME_DISTANCE])
            visual_similar_frame = str(int(correct_frame_a) + random_number)
            images = task_data.image_paths.get(visual_similar_frame, {})
            if sensor_name in images:
                distractors.append(images[sensor_name])
    

        # Strategy 3: Get randomly sampled frames
        remaining_candidates = [img for img in candidate_images if img not in distractors]
        random.shuffle(remaining_candidates)
        
        while len(distractors) < 3 and remaining_candidates:
            candidate_frame = remaining_candidates.pop()
            candidate_frame_id = candidate_frame.split('/')[-1].split('.')[0]
            if int(candidate_frame_id) <= task_data.scene_graph_reader.get_frame_number():
                new_candidate_frame_id = int(candidate_frame_id) + 30
                candidate_frame = candidate_frame.split('/')[:-1] + [f"{new_candidate_frame_id:05d}.png"]
                candidate_frame = '/'.join(candidate_frame)
            distractors.append(candidate_frame)

        return distractors[:3]

    def _generate_advanced_distractors(self, task_data: TaskData, correct_frame_a: str,
                                     ground_truth_diff: Dict[str, Any],
                                     available_frames: List[str], sensor_name: str) -> List[str]:
        """
        Generate semantically plausible but incorrect distractor images.
        
        This implements the advanced strategy for creating fake state changes.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame
            ground_truth_diff: The true diff
            available_frames: Available frames to choose from
            sensor_name: Sensor name to use
            
        Returns:
            List[str]: Advanced distractor image paths
        """
        advanced_distractors = []
        
        try:
            # Strategy: Create "fake" diffs that are plausible variations of the ground truth
            fake_diffs = self._generate_fake_diffs(ground_truth_diff)
            
            for fake_diff in fake_diffs:
                if len(advanced_distractors) >= 2:  # Limit advanced distractors
                    break
                
                # Find a frame whose scene graph most closely matches what the fake diff would produce
                best_match_frame = self._find_best_matching_frame(
                    task_data, correct_frame_a, fake_diff, available_frames
                )
                print(f"Best match frame: {best_match_frame}")
                print(f"available frames: {available_frames}")
                if best_match_frame:
                    candidate_frame = int(best_match_frame) + 20
                    if candidate_frame <= task_data.scene_graph_reader.get_frame_number():
                        best_match_frame = str(candidate_frame)
                    images = task_data.image_paths.get(best_match_frame, {})
                    if sensor_name in images:
                        distractor_image = images[sensor_name]
                        if distractor_image not in advanced_distractors:
                            advanced_distractors.append(distractor_image)
                            
        except Exception as e:
            print(f"Error generating advanced distractors: {e}")
        
        return advanced_distractors
    
    def _generate_fake_diffs(self, ground_truth_diff: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate plausible variations of the ground truth diff.
        
        Args:
            ground_truth_diff: The original diff
            
        Returns:
            List[Dict]: List of fake diffs
        """
        fake_diffs = []
        
        # Strategy 1: Modify target objects in relations
        for operation in ['add', 'remove', 'update']:
            if operation in ground_truth_diff:
                for edge in ground_truth_diff[operation].get('edges', []):
                    # Create a variation where the same object interacts with a different target
                    fake_diff = copy.deepcopy(ground_truth_diff)
                    
                    # Modify the 'to' object in the edge (keep 'from' and 'states' the same)
                    fake_edge = fake_diff[operation]['edges'][0]  # Simplification: modify first edge
                    
                    # Generate a plausible alternative target
                    alternative_targets = self._get_alternative_targets(edge)
                    if alternative_targets:
                        fake_edge['to'] = random.choice(alternative_targets)
                        fake_diffs.append(fake_diff)
        
        # Strategy 2: Modify the operation type (add -> remove, etc.)
        if len(fake_diffs) < 2:
            operation_mappings = {
                'add': 'remove',
                'remove': 'add', 
                'update': 'add'  # update -> add (different end state)
            }
            
            for orig_op, new_op in operation_mappings.items():
                if orig_op in ground_truth_diff and len(fake_diffs) < 2:
                    fake_diff = copy.deepcopy(ground_truth_diff)
                    
                    # Move changes from original operation to new operation
                    if new_op not in fake_diff:
                        fake_diff[new_op] = {'nodes': [], 'edges': []}
                    
                    fake_diff[new_op] = fake_diff[orig_op]
                    del fake_diff[orig_op]
                    fake_diffs.append(fake_diff)
        
        return fake_diffs[:2]  # Return at most 2 fake diffs
    
    def _get_alternative_targets(self, edge: Dict[str, Any]) -> List[str]:
        """
        Get alternative target objects for a relation.
        
        Args:
            edge: Original edge data
            
        Returns:
            List[str]: Alternative target object names
        """
        # Simple heuristic: generate common object names that could be targets
        common_objects = [
            'table_1', 'counter_1', 'shelf_1', 'floor_1', 'chair_1',
            'bowl_1', 'plate_1', 'cup_1', 'fridge_1', 'sink_1'
        ]
        
        original_target = edge.get('to', '')
        alternatives = [obj for obj in common_objects if obj != original_target]
        
        return alternatives[:3]  # Return up to 3 alternatives
    
    def _find_best_matching_frame(self, task_data: TaskData, base_frame: str,
                                 fake_diff: Dict[str, Any], available_frames: List[str]) -> str:
        """
        Find the frame whose scene graph best matches the result of applying fake_diff to base_frame.
        
        Args:
            task_data: Task data
            base_frame: Base frame to apply diff to
            fake_diff: The fake diff to apply
            available_frames: Available frames to search
            
        Returns:
            str: Best matching frame ID, or None if no good match
        """
        try:
            # Get the base scene graph
            base_graph = task_data.scene_graph_reader.get_scene_graph(base_frame)
            
            # Apply the fake diff to get the target scene graph
            target_graph = self._apply_diff_to_graph(base_graph, fake_diff)
            
            # Find the frame with the most similar scene graph
            best_match = None
            best_similarity = 0
            
            for frame_id in available_frames:
                try:
                    frame_graph = task_data.scene_graph_reader.get_scene_graph(frame_id)
                    similarity = self._compute_graph_similarity(target_graph, frame_graph)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = frame_id
                        
                except Exception:
                    continue
            
            # Only return a match if similarity is above a threshold
            if best_similarity > 0.5:  # Threshold for reasonable similarity
                return best_match
                
        except Exception as e:
            print(f"Error finding best matching frame: {e}")
        
        return None
    
    def _apply_diff_to_graph(self, base_graph: Dict[str, List], diff: Dict[str, Any]) -> Dict[str, List]:
        """
        Apply a diff to a scene graph (simplified implementation).
        
        Args:
            base_graph: Base scene graph
            diff: Diff to apply
            
        Returns:
            Dict: Modified scene graph
        """
        # This is a simplified implementation
        # In practice, this would use the same logic as SceneGraphReader._apply_diff_to_graph
        result_graph = copy.deepcopy(base_graph)
        
        # Simplified application - just modify the first relevant nodes/edges
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # This is a placeholder - in practice would need full diff application logic
                pass
        
        return result_graph
    
    def _compute_graph_similarity(self, graph1: Dict[str, List], graph2: Dict[str, List]) -> float:
        """
        Compute similarity between two scene graphs.
        
        Args:
            graph1: First scene graph
            graph2: Second scene graph
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple similarity based on shared nodes and edges
        nodes1 = {node.get('name') for node in graph1.get('nodes', [])}
        nodes2 = {node.get('name') for node in graph2.get('nodes', [])}
        
        edges1 = {(edge.get('from'), edge.get('to')) for edge in graph1.get('edges', [])}
        edges2 = {(edge.get('from'), edge.get('to')) for edge in graph2.get('edges', [])}
        
        # Jaccard similarity
        node_intersection = len(nodes1 & nodes2)
        node_union = len(nodes1 | nodes2)
        node_similarity = node_intersection / node_union if node_union > 0 else 0
        
        edge_intersection = len(edges1 & edges2)
        edge_union = len(edges1 | edges2)
        edge_similarity = edge_intersection / edge_union if edge_union > 0 else 0
        
        # Weighted average
        return 0.6 * node_similarity + 0.4 * edge_similarity
    
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
    

class MultiStepForwardDynamicsGenerator(AbstractQAGenerator):
    """
    Generates multi-step forward dynamics QA pairs.

    Multi-Step Forward Dynamics: Given the current state and a sequence of actions, what will be the next following states?
    """

    def __init__(self, qa_gen_logic: str = "multi-choice", visual_prompt: bool=True, step_length: int=5, option_num: int=4):
        """
        Initialize the forward dynamics generator.

        Args:
            qa_gen_logic: Optional logic specifical.
        """
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
        self.translator = StateChangeTranslator(
            type='multi_forward_dynamics'
        )
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 30, f"Step length should be between 2 and 30. Got {self.step_length} instead."
        self.sensor_names = ['external_sensor1']
        self.option_num = option_num
        assert self.option_num >= 4, f"Option number should be at least 4. Got {self.option_num} instead."
    
    @property
    def qa_type(self) -> str:
        if self.qa_gen_logic == "ordering":
            return f"forward_dynamics_ordering_{self.step_length}_steps"
        elif self.qa_gen_logic == "multi-choice" or self.qa_gen_logic == None:
            return f"forward_dynamics_option_{self.step_length}_steps_{self.option_num}_choices"
        else:
            raise ValueError(f"Invalid QA generation logic: {self.qa_gen_logic}")
    
    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Path to the visual prompt for this QA generator. Should be default to QA_images/[qa_type]/[images]

        Returns:
            str: Path to the visual prompt
        """
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


    def generate(self, task_data: TaskData, num_to_sample: int=30, max_qa_num: int=25) -> List[QAPair]: # Should be List[QAPair]
        """
        Generates all valid sequences of frames of length `self.step_length`.

        A sequence [f1, f2, ..., fk] is valid if every consecutive pair 
        (fi, f_{i+1}) meets the state change criteria.

        Args:
            task_data: Task data containing scene graphs and images.
            num_to_sample: Number of sequences to sample.
            max_qa_num: Maximum number of QA pairs to generate.
        Returns:
            List[QAPair]: List of generated QA pairs.
        """
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
        
        # Step 3: Weighted Random Sampling
        # Ensure we don't try to sample more paths than exist
        actual_num_to_sample = min(num_to_sample, total_paths)
        all_valid_sequences = self._sample_paths_randomly(
            actual_num_to_sample, graph, dp_table, key_frame_ids
        )
        
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")
        
        qa_pairs = []
        print(f"Phase 4: Generating QA pairs from {len(all_valid_sequences)} sequences...")
        if mode == "multi-choice":
            for seq in tqdm(all_valid_sequences, desc="Generating Q&A"):
                try:
                    # Generate distractors for the current correct sequence
                    distractor_sequences = self._generate_distractor_sequences(
                        seq, all_valid_sequences, task_data,
                        graph=graph, dp_table=dp_table, key_frame_ids=key_frame_ids,
                        frame_to_index=frame_to_index
                    )
                    
                    if len(distractor_sequences) < self.option_num - 1:
                        continue

                    # Create the QA pair
                    qa_pair = self._create_multistep_qa_pair(task_data, seq, distractor_sequences)
                    
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    import traceback
                    print(f"Error generating QA for sequence {seq}: {e}")
                    traceback.print_exc() # Uncomment for detailed debugging
                    continue
        elif mode == "ordering":
            for seq in tqdm(all_valid_sequences, desc="Generating Ordering QA pairs"):
                try:
                    qa_pair = self._create_ordering_qa_pair(task_data, seq)
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    import traceback
                    print(f"Error generating QA for sequence {seq}: {e}")
                    traceback.print_exc() # Uncomment for detailed debugging
                    continue
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print(f"\nGenerated {len(qa_pairs)} multi-step forward dynamics QA pairs.")
        if max_qa_num:
            print(f"Truncating to {max_qa_num} QA pairs.")
            # do random sampling
            qa_pairs = random.sample(qa_pairs, min(max_qa_num, len(qa_pairs)))
        return qa_pairs
    

    def get_valid_path_num(self, task_data: TaskData, step_num: int) -> int:
        """
        Get the number of valid paths of a given length.
        """
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        frame_to_index = {frame_id: i for i, frame_id in enumerate(key_frame_ids)}
        dp_table = self._count_paths_with_dp(graph, key_frame_ids, frame_to_index)
        return dp_table[step_num - 1].sum()
    
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
        Create a QA pair for the ordering task.
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        # 1. Get the initial state image
        start_frame_id = correct_sequence[0]
        sensor_name = self.sensor_names[0]
        start_image_path = task_data.image_paths.get(start_frame_id, {}).get(sensor_name, None)
        if not start_image_path:
            return None
        
        # 2. Generate the question from the action sequence
        action_descriptions = self._translate_sequence_to_actions(task_data, correct_sequence)
        
        # 3. Get the next states (excluding the initial state)
        next_states = correct_sequence[1:]
        
        # 4. Create shuffled ordering and track the correct order
        shuffled_next_states = next_states[:]
        random.shuffle(shuffled_next_states)
        
        # 5. Find the correct order - map from shuffled positions to original positions
        correct_order = []
        for original_frame in next_states:
            shuffled_position = shuffled_next_states.index(original_frame) + 1  # 1-indexed
            correct_order.append(shuffled_position)
        
        # 6. Generate QA pair ID
        qa_id = self.generate_qa_id_hash(task_data.task_name, self.qa_type, correct_sequence)
        
        # 7. Create visual prompts by reusing existing infrastructure
        final_start_image = start_image_path
        final_next_state_images = []
        
        if self.visual_prompt:
            # Reuse existing visual prompt creation by adapting it for ordering
            # We need individual labeled images, not filmstrips
            image_root_dir = task_data.image_root_path.parent
            new_base_dir = self.visual_prompt_path(image_root_dir)
            output_dir = Path(new_base_dir) / task_data.task_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process current state image using existing method
            cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
            self._add_text_to_image(start_image_path, "Current State", str(cur_state_output_path))
            final_start_image = str(cur_state_output_path)
            
            # Process each shuffled next state image with sequential labels using existing method
            for i, frame_id in enumerate(shuffled_next_states):
                next_state_image_path = task_data.image_paths[frame_id][sensor_name]
                label = f"Next State {i + 1}"
                next_state_output_path = output_dir / f"{qa_id}_next_state_{i + 1}.png"
                self._add_text_to_image(next_state_image_path, label, str(next_state_output_path))
                final_next_state_images.append(str(next_state_output_path))
        else:
            # If no visual prompt, use original images
            for frame_id in shuffled_next_states:
                image_path = task_data.image_paths[frame_id][sensor_name]
                final_next_state_images.append(image_path)
        
        # 8. Generate the question using the ordering prompt
        question = multi_fwd_ordering_prompt.format(STATE_CHANGES=action_descriptions)
        
        # 9. Prepare all images (current state + shuffled next states)
        all_images = [final_start_image] + final_next_state_images
        # remove the root dir from the image paths
        all_images = [str(Path(image_path).relative_to(image_root_dir)) for image_path in all_images]
        # 10. Create the ground truth answer
        gt_answer = {
            "type": f"{self.qa_type}",
            "options": [],
            'correct_option': correct_order
            # "shuffled_frames": shuffled_next_states,  # For debugging/reference
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
            
            # if task_data.scene_graph_reader.has_similar_edges(grounded_visible_diff, candidate_visible_diff, grounded_current_scene_graph, grounded_next_scene_graph, candidate_scene_graph):
            #     return False
            
            # check if grounded_visible_diff is a subset of candidate_visible_diff
            if not task_data.scene_graph_reader.is_diff_subset_scene(grounded_visible_diff, candidate_scene_graph) and not task_data.scene_graph_reader.is_diff_subset_scene(candidate_visible_diff, grounded_next_scene_graph):
                all_subsets = False
        
        return not all_subsets
    
    def _search_top_k_similar_sequences(self, correct_sequence: List[str], all_valid_sequences: List[List[str]], top_k: int=5) -> List[List[str]]:
        """
        Search for the top `top_k` most similar sequences to the correct sequence.
        
        Similarity is measured by:
        1. Number of different frame IDs (fewer = more similar)
        2. Sum of distances between different IDs (smaller = more similar) 
        3. Random order for ties
        
        Args:
            correct_sequence: The ground truth sequence to compare against
            all_valid_sequences: Pool of candidate sequences to search from
            top_k: Number of most similar sequences to return
            
        Returns:
            List of the top k most similar sequences
        """
        if not all_valid_sequences:
            return []
        
        # Filter out sequences with different lengths and the correct sequence itself
        candidates = [seq for seq in all_valid_sequences 
                     if len(seq) == len(correct_sequence) and seq != correct_sequence]
        
        if not candidates:
            return []
        
        def compute_similarity_score(candidate_seq):
            """Compute similarity score for a candidate sequence"""
            num_differences = 0
            total_distance = 0
            
            for i in range(len(correct_sequence)):
                correct_id = int(correct_sequence[i])
                candidate_id = int(candidate_seq[i])
                
                if correct_id != candidate_id:
                    num_differences += 1
                    total_distance += abs(correct_id - candidate_id)
            
            # Return tuple for sorting: (num_differences, total_distance, random_tie_breaker)
            return (num_differences, total_distance, random.random())
        
        # Compute scores for all candidates
        scored_candidates = [(seq, compute_similarity_score(seq)) for seq in candidates]
        
        # Sort by similarity score (ascending: fewer differences, smaller distances, random)
        scored_candidates.sort(key=lambda x: x[1])
        
        # Return top k sequences
        return [seq for seq, _ in scored_candidates[:top_k]]
    
    def _sort_similar_sequences(self, correct_sequence: List[str], all_valid_sequences: List[List[str]]) -> List[List[str]]:
        """
        Sort the sequences by similarity to the correct sequence.
        """
        similar_sequences = self._search_top_k_similar_sequences(correct_sequence, all_valid_sequences, len(all_valid_sequences))
        return similar_sequences
    
    def _generate_hard_negatives_by_modification(
        self,
        correct_sequence: List[str],
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str],
        frame_to_index: Dict[str, int],
        num_to_generate: int,
        task_data: TaskData
    ) -> List[List[str]]:
        """
        Generates "hard negative" distractors with a complete 3-phase hybrid strategy.
        This version ensures MAXIMUM DIVERSITY in fallback phases by first collecting all
        possible valid swaps, shuffling them, and then selecting the required number.
        """
        k = self.step_length
        generated_distractors: Set[tuple] = {tuple(correct_sequence)}
        out: List[List[str]] = []
        
        max_diff = min(2, k // 4)
        if max_diff <= 0 or num_to_generate == 0:
            return []

        # --- Phase 1: Randomized Search (Fastest method) ---
        # This phase is inherently diverse and remains unchanged.
        max_attempts = num_to_generate * 20
        for _ in range(max_attempts):
            if len(out) >= num_to_generate: break
            
            temp_sequence = list(correct_sequence)
            num_swaps = random.randint(1, max_diff)
            
            if k < 3 or k - 2 < num_swaps: continue
            
            indices_to_modify = random.sample(range(1, k - 1), k=num_swaps)
            
            successful_modification = True
            for i in indices_to_modify:
                prev_frame = temp_sequence[i - 1]
                next_frame = temp_sequence[i + 1]
                candidates = [
                    f for f in graph.get(prev_frame, [])
                    if f != temp_sequence[i]
                    and dp_table[i, frame_to_index[f]] > 0
                    and next_frame in graph.get(f, [])
                ]
                if not candidates:
                    successful_modification = False
                    break
                temp_sequence[i] = random.choice(candidates)

            if successful_modification:
                is_valid_path = all(b in graph.get(a, []) for a, b in zip(temp_sequence, temp_sequence[1:]))
                if is_valid_path and self._validate_not_all_subsets(task_data, correct_sequence, temp_sequence):
                    t_seq = tuple(temp_sequence)
                    if t_seq not in generated_distractors:
                        generated_distractors.add(t_seq)
                        out.append(temp_sequence)

        if len(out) >= num_to_generate:
            return out[:num_to_generate]

        # --- Phase 2: 1-Point Fallback (MAXIMUM DIVERSITY) ---
        need = num_to_generate - len(out)
        
        # Step 1: Collect ALL possible and valid 1-point swaps
        all_possible_1_swaps = []
        for i in range(1, k - 1):
            prev, orig, nxt = correct_sequence[i - 1], correct_sequence[i], correct_sequence[i + 1]
            
            if i > 0 and dp_table[i - 1, frame_to_index[prev]] == 0: continue
            
            possible_cands = [
                f for f_idx, count in enumerate(dp_table[i])
                if count > 0 and (f := key_frame_ids[f_idx]) != orig
            ]
            for cand in possible_cands:
                if cand in graph.get(prev, []) and nxt in graph.get(cand, []):
                    new_seq = correct_sequence[:i] + [cand] + correct_sequence[i + 1:]
                    if self._validate_not_all_subsets(task_data, correct_sequence, new_seq):
                        all_possible_1_swaps.append(new_seq)

        # Step 2: Shuffle the collected pool
        random.shuffle(all_possible_1_swaps)
        
        # Step 3: Add from the shuffled pool until `need` is met
        for new_seq in all_possible_1_swaps:
            if need <= 0: break
            t_seq = tuple(new_seq)
            if t_seq not in generated_distractors:
                generated_distractors.add(t_seq)
                out.append(new_seq)
                need -= 1

        if need <= 0 or max_diff < 2:
            return out[:num_to_generate]

        # --- Phase 3: 2-Point Fallback (MAXIMUM DIVERSITY) ---
        if k < 4: return out[:num_to_generate]
        
        # Step 1: Collect ALL possible and valid 2-point swaps
        all_possible_2_swaps = []
        for i, j in itertools.combinations(range(1, k - 1), 2):
            prev_i, orig_i = correct_sequence[i - 1], correct_sequence[i]
            orig_j, next_j = correct_sequence[j], correct_sequence[j + 1]

            if i > 0 and dp_table[i - 1, frame_to_index[prev_i]] == 0: continue
            
            cands_i = [f for f_idx, count in enumerate(dp_table[i]) if count > 0 and (f := key_frame_ids[f_idx]) != orig_i and f in graph.get(prev_i, [])]

            for cand_i in cands_i:
                is_adjacent = (j == i + 1)
                prev_j = cand_i if is_adjacent else correct_sequence[j - 1]
                if not is_adjacent and correct_sequence[i + 1] not in graph.get(cand_i, []): continue
                if dp_table[j - 1, frame_to_index[prev_j]] == 0: continue
                
                cands_j = [f for f_idx, count in enumerate(dp_table[j]) if count > 0 and (f := key_frame_ids[f_idx]) != orig_j and f in graph.get(prev_j, []) and next_j in graph.get(f, [])]

                for cand_j in cands_j:
                    new_seq = list(correct_sequence)
                    new_seq[i], new_seq[j] = cand_i, cand_j
                    if self._validate_not_all_subsets(task_data, correct_sequence, new_seq):
                        all_possible_2_swaps.append(new_seq)
        
        # Step 2: Shuffle the collected pool
        random.shuffle(all_possible_2_swaps)
        
        # Step 3: Add from the shuffled pool until `need` is met
        for new_seq in all_possible_2_swaps:
            if need <= 0: break
            t_seq = tuple(new_seq)
            if t_seq not in generated_distractors:
                generated_distractors.add(t_seq)
                out.append(new_seq)
                need -= 1

        return out[:num_to_generate]

    def _generate_distractor_sequences(
        self,
        correct_sequence: List[str],
        all_valid_sequences: List[List[str]],
        task_data: TaskData,
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str],
        frame_to_index: Dict[str, int]
    ) -> List[List[str]]:
        """
        Generates `self.option_num - 1` distractor sequences based on the defined heuristics.
        """
        distractors = []
        # Create a pool of candidates to avoid reusing the correct sequence
        candidate_pool = [seq for seq in all_valid_sequences if seq != correct_sequence]
        random.shuffle(candidate_pool)

        similar_sequences = self._sort_similar_sequences(correct_sequence, candidate_pool)

        # Heuristic 1: Get similar sequences, always try to get 1/3 of the distractors
        # Try to find a sequence that shares the first step but diverges.
        num_hard_negatives_to_gen = (self.option_num - 1) // 3 * 3
        if len(correct_sequence) > 2:
            hard_negatives = self._generate_hard_negatives_by_modification(
                correct_sequence,
                graph,
                dp_table,
                key_frame_ids,
                frame_to_index,
                num_hard_negatives_to_gen,
                task_data
            )
            distractors.extend(hard_negatives)

            # Remove the newly generated distractors from the candidate_pool to avoid duplicates
            # Using a set for efficient lookup
            hard_negatives_set = {tuple(seq) for seq in hard_negatives}
            candidate_pool = [seq for seq in candidate_pool if tuple(seq) not in hard_negatives_set]

        # if len(correct_sequence) > 2:
        #     searched_num = (self.option_num-1) // 3
        #     cur_num = 0
        #     for seq in similar_sequences:
        #         if cur_num >= searched_num:
        #             break
        #         if self._validate_not_all_subsets(task_data, correct_sequence, seq):
        #             distractors.append(seq)
        #             candidate_pool.remove(seq)
        #             cur_num += 1

        # Heuristic 2 (New): Enumerate all single-pair-swapped sequences
        num_incorrect_order_to_gen = self.option_num - 1 - len(distractors)
        cur_num = 0
        
        answer_part = correct_sequence[1:]
        if len(answer_part) >= 2:
            # Generate all unique pairs of indices to swap, e.g., (0, 1), (0, 2), (1, 2), ...
            for i, j in itertools.combinations(range(len(answer_part)), 2):
                if cur_num >= num_incorrect_order_to_gen:
                    break

                # Create the new sequence by swapping the pair
                permuted_part = answer_part[:]
                permuted_part[i], permuted_part[j] = permuted_part[j], permuted_part[i]
                distractor_candidate = [correct_sequence[0]] + permuted_part
                
                # Perform the same validity checks as before
                # Using a set for the main check is more efficient
                distractor_set = {tuple(d) for d in distractors}
                if (tuple(distractor_candidate) not in distractor_set and
                    self._validate_not_all_subsets(task_data, correct_sequence, distractor_candidate)):
                    
                    distractors.append(distractor_candidate)
                    # It's safer to create a new pool after modifications or filter it at the end
                    # but for simplicity, we keep the remove() operation as in the original
                    candidate_pool.remove(distractor_candidate) if distractor_candidate in candidate_pool else None
                    cur_num += 1

        # Heuristic 3: Partial Execution (Easy), another 1/3 of the distractors
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) - len(distractors)
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random position to modify
                modify_positions = list(range(len(correct_sequence)))
                if not modify_positions:
                    break
                
                modify_pos = random.choice(modify_positions)
                
                # Choose a random state from the sequence to repeat at the modify position
                repeat_state = random.choice(correct_sequence)
                
                # Create sequence with repeated state
                partial_execution_seq = correct_sequence[:]
                partial_execution_seq[modify_pos] = repeat_state
                
                # Make sure we actually created a different sequence
                if partial_execution_seq == correct_sequence:
                    continue
                
                # Check if this distractor is valid and unique
                if (partial_execution_seq not in distractors and 
                    partial_execution_seq != correct_sequence and
                    self._validate_not_all_subsets(task_data, correct_sequence, partial_execution_seq)):
                    
                    distractors.append(partial_execution_seq)
                    candidate_pool.remove(partial_execution_seq) if partial_execution_seq in candidate_pool else None
                    cur_num += 1

        # Heuristic 4: Fill with random valid sequences (Fallback)
        while len(distractors) < self.option_num - 1 and candidate_pool:
            distractor = candidate_pool.pop()
            if distractor not in distractors and self._validate_not_all_subsets(task_data, correct_sequence, distractor):
                distractors.append(distractor)
        
        # If we still don't have enough, we might need simpler fallbacks,
        # but for now, this should be sufficient.
        return distractors[:self.option_num - 1]
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> str:
        """
        Translates a sequence of frame IDs into a single, chained action description.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            # Use the translator you already have
            action_desc = self.translator.translate_diff(diff)
            action_descriptions.append(action_desc)

        # Combine descriptions into a numbered sequence
        if not action_descriptions:
            raise ValueError("No actions are performed.")
        
        # Format: "First, [action1]. Then, [action2]. Finally, [action3]."
        formatted_actions = []
        action_template = "[Action {i}] {action}"
        for i, desc in enumerate(action_descriptions):
            action = action_template.format(i=i+1, action=desc)
            formatted_actions.append(action)
            
        return "\n".join(formatted_actions)
    
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

    def _draw_text_on_image(self, image: Image.Image, text: str) -> Image.Image:
        """Helper function to draw a styled label onto a PIL Image object."""
        # (This is the same helper function as defined above)
        draw = ImageDraw.Draw(image)
        font_size = max(30, image.height // 12)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text_color, outline_color, outline_width, x, y = (255, 20, 20), (255, 255, 255), 2, 15, 15
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
        frame_labels: List[str],
        overall_label: str
    ) -> None:
        """
        Stitches images into a filmstrip, adding a header for the overall label
        and labeling each frame individually to prevent text overlap.
        """
        if len(image_paths) != len(frame_labels):
            raise ValueError("The number of image paths and frame labels must be equal.")
            
        try:
            # First, create the individually labeled frames
            labeled_images = []
            for i, path in enumerate(image_paths):
                img = Image.open(path)
                # Draw "Next State X" on each frame
                labeled_img = self._draw_text_on_image(img, frame_labels[i])
                labeled_images.append(labeled_img)

            # Now, prepare the final filmstrip with a header
            images = labeled_images
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            header_height = 60  # The height of the new top banner for the "Option" label

            # Create the final canvas with space for the header
            filmstrip = Image.new('RGB', (total_width, max_height + header_height), (255, 255, 255))
            
            # Draw the overall label (e.g., "Option A") in the header space
            # We use a slightly larger font for the main option label
            draw = ImageDraw.Draw(filmstrip)
            font_size = 40
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

            draw.text((15, 10), overall_label, font=font, fill=(0, 0, 0)) # Black text for the header

            # Paste the labeled frames below the header
            x_offset = 0
            for img in images:
                filmstrip.paste(img, (x_offset, header_height)) # Use y-offset for the header
                x_offset += img.size[0]
            
            filmstrip.save(output_path)
        except Exception as e:
            print(f"Error creating labeled filmstrip for {output_path}: {e}")
            
    def _create_visual_prompt_for_sequences(
        self,
        qa_id: str,
        cur_state_image: str,
        all_options_sequences: List[List[str]], # This takes sequences of frame IDs
        task_data: TaskData
    ) -> Tuple[str, List[str]]:
        """
        Creates and saves all visual components for a multi-step QA pair.
        This includes the labeled "Current State" image and the labeled "filmstrip"
        images for each answer option.

        Args:
            qa_id: The unique ID for this question-answer pair.
            cur_state_image: Path to the initial state image.
            all_options_sequences: A list containing 4 items, where each item is a
                                   sequence of frame IDs representing an answer option.
            task_data: The task data object.

        Returns:
            A tuple containing:
            - The path to the newly created and labeled "Current State" image.
            - A list of paths to the newly created "filmstrip" option images.
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        output_dir = Path(new_base_dir) / task_data.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        sensor_name = self.sensor_names[0]

        # 1. Process and label the "Current State" image using the original _add_text_to_image method
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))
        
        # 2. Process and create a labeled filmstrip for each option
        option_labels = [f"Option {chr(65 + i)}" for i in range(len(all_options_sequences))]

        option_output_paths = []

        for i, frame_id_seq in enumerate(all_options_sequences):
            # Get the actual image paths for the sequence of frame IDs
            try:
                image_paths_for_seq = [
                    task_data.image_paths[frame_id][sensor_name] for frame_id in frame_id_seq
                ]
            except KeyError:
                print(f"Warning: Could not find image for a frame in sequence {frame_id_seq}. Skipping option.")
                continue

            # Define the output path for the filmstrip
            filmstrip_path = output_dir / f"{qa_id}_option_{chr(65 + i)}.png"
            
            # --- Generate labels for each frame in the option sequence ---
            num_frames = len(image_paths_for_seq)
            frame_labels = [f"Next State {j+1}" for j in range(num_frames)]
            # --- End of new logic ---

            self._create_filmstrip_image(
                image_paths=image_paths_for_seq,
                output_path=str(filmstrip_path),
                frame_labels=frame_labels,
                overall_label=option_labels[i] # Pass the overall label
            )
            option_output_paths.append(str(filmstrip_path))

        return str(cur_state_output_path), option_output_paths
    
    def _create_multistep_qa_pair(
        self,
        task_data: TaskData,
        correct_sequence: List[str],
        distractor_sequences: List[List[str]]
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step forward dynamics question.
        """
        if len(distractor_sequences) < 3:
            return None

        # 1. Get the initial state image
        start_frame_id = correct_sequence[0]
        sensor_name = self.sensor_names[0]
        start_image_path = task_data.image_paths.get(start_frame_id, {}).get(sensor_name)
        if not start_image_path:
            return None

        # 2. Generate the question from the action sequence
        action_description = self._translate_sequence_to_actions(task_data, correct_sequence)
        # You can define `multi_fwd_prompt` in your prompts file, e.g.:
        # multi_fwd_prompt = "Given the current state, if you perform the following actions in order, what will the sequence of resulting states look like?\nActions: {STATE_CHANGES}"
        question = multi_fwd_prompt.format(STATE_CHANGES=action_description)

        # 3. Prepare options (correct answer + distractors)
        # An option is a sequence of frame IDs, representing the states *after* the initial one.
        correct_option_seq = correct_sequence[1:]
        all_options_seqs = [correct_option_seq] + [d[1:] for d in distractor_sequences]
        
        # Shuffle and find the correct answer's new index
        random.shuffle(all_options_seqs)
        try:
            correct_option_index = all_options_seqs.index(correct_option_seq)
        except ValueError:
            return None # Should not happen if logic is correct

        # 4. Generate the QA pair ID
        qa_id = f"{task_data.task_name}_{self.qa_type}_{'_'.join(correct_sequence)}"
        
        final_start_image = start_image_path
        final_option_images = []

        # 5. Create and save the visual prompt for the images if enabled
        if self.visual_prompt:
            final_start_image, final_option_images = self._create_visual_prompt_for_sequences(
                qa_id=qa_id,
                cur_state_image=start_image_path,
                all_options_sequences=all_options_seqs,
                task_data=task_data
            )
        else:
            # If not creating visual prompts, the options will be a list of lists of original image paths
            for frame_id_seq in all_options_seqs:
                image_paths_for_seq = [
                    task_data.image_paths[frame_id][sensor_name] for frame_id in frame_id_seq
                ]
                final_option_images.append(image_paths_for_seq)

        # 6. Assemble the final QAPair
        gt_answer = {
            "type": self.qa_type,
            "options": final_option_images, # Paths to filmstrips or list of lists of paths
            "correct_option": chr(correct_option_index + 65),
        }

        images = [final_start_image] + final_option_images
        
        qa_pair = QAPair(
            id=qa_id,
            images=images,
            task_name=task_data.task_name,
            key_frame_ids=correct_sequence,
            question=question,
            gt_answer=gt_answer
        )
        return qa_pair


class MultiStepForwardDynamicsAblationGenerator(AbstractQAGenerator):
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
            type='multi_forward_dynamics'
        )
        self.qa_gen_logic = 'ordering'
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 30, f"Step length should be between 2 and 30. Got {self.step_length} instead."
        self.sensor_names = ['external_sensor1']
    
    @property
    def qa_type(self) -> str:
        return f"forward_dynamics_ordering_{self.step_length}_steps"
    
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

    def _create_no_image_ordering_qa_pair(
            self, 
            task_data: TaskData, 
            correct_sequence: List[str]
        ) -> QAPair:
        """
        Create a QA pair for the ordering task.
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        # 1. Get the initial state image
        start_frame_id = correct_sequence[0]
        sensor_name = self.sensor_names[0]
        start_image_path = task_data.image_paths.get(start_frame_id, {}).get(sensor_name, None)
        if not start_image_path:
            return None
        
        # 2. Generate the question from the action sequence
        action_descriptions = self._translate_sequence_to_actions(task_data, correct_sequence)
        
        # 3. Get the next states (excluding the initial state)
        next_states = correct_sequence[1:]
        
        # 4. Create shuffled ordering and track the correct order
        shuffled_next_states = next_states[:]
        random.shuffle(shuffled_next_states)
        
        # 5. Find the correct order - map from shuffled positions to original positions
        correct_order = []
        for original_frame in next_states:
            shuffled_position = shuffled_next_states.index(original_frame) + 1  # 1-indexed
            correct_order.append(shuffled_position)
        
        step_num = len(correct_sequence)
        qa_type = f"forward_dynamics_ordering_{step_num}_steps"
        # 6. Generate QA pair ID
        qa_id = self.generate_qa_id_hash(task_data.task_name, qa_type, correct_sequence)
        

        # 8. Generate the question using the ordering prompt
        question = multi_fwd_ordering_prompt.format(STATE_CHANGES=action_descriptions)

        # 10. Create the ground truth answer
        gt_answer = {
            "type": f"{qa_type}",
            "options": [],
            'correct_option': correct_order
            # "shuffled_frames": shuffled_next_states,  # For debugging/reference
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

    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> str:
        """
        Translates a sequence of frame IDs into a single, chained action description.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            # Use the translator you already have
            action_desc = self.translator.translate_diff(diff)
            action_descriptions.append(action_desc)

        # Combine descriptions into a numbered sequence
        if not action_descriptions:
            raise ValueError("No actions are performed.")
        
        # Format: "First, [action1]. Then, [action2]. Finally, [action3]."
        formatted_actions = []
        action_template = "[Action {i}] {action}"
        for i, desc in enumerate(action_descriptions):
            action = action_template.format(i=i+1, action=desc)
            formatted_actions.append(action)
            
        return "\n".join(formatted_actions)
    
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
            
            # Save the image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error adding text to image {image_path}: {e}")
            # If text addition fails, just copy the original image
            import shutil
            shutil.copy2(image_path, output_path)
    
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
                
            # Create output structure for this ablation - separate folder per ablation
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
                    
                    # 2. Next state images (remaining frames) - shuffled to match gt_answer order
                    next_frame_ids = key_frame_ids[1:]
                    gt_answer = qa_data['gt_answer']
                    
                    # Create shuffled order based on gt_answer (reverse mapping)
                    shuffled_next_states = [''] * len(next_frame_ids)
                    for original_idx, shuffled_pos in enumerate(gt_answer):
                        shuffled_next_states[shuffled_pos - 1] = next_frame_ids[original_idx]  # shuffled_pos is 1-indexed
                    
                    for i, frame_id in enumerate(shuffled_next_states):
                        if not frame_id:  # Skip if empty
                            continue
                            
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
        qa_output_file = qa_output_dir / "forward_behavior_eqa_ordering.jsonl"
        
        with open(qa_output_file, "w") as f:
            for qa_pair in all_qa_pairs_with_images:
                f.write(json.dumps(qa_pair) + "\n")
        
        print(f"\nSaved {len(all_qa_pairs_with_images)} total QA pairs from {len(ablation_folders)} ablation settings to {qa_output_file}")
        
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
    raw_scene_graph_file = raw_data_dir / "raw_scene_graph_0.json"
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

    file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/original_ordering_samples.jsonl"

    
    # ablation step 1 generation
    # step_3_generator = MultiStepForwardDynamicsAblationGenerator(step_length=3)
    # step_3_qa_pairs = step_3_generator.generate_candidates(task_data)

    # step_6_generator = MultiStepForwardDynamicsAblationGenerator(step_length=6)
    # step_6_qa_pairs = step_6_generator.generate_candidates(task_data)

    # step_9_generator = MultiStepForwardDynamicsAblationGenerator(step_length=9)
    # step_9_qa_pairs = step_9_generator.generate_candidates(task_data)

    # all_qa_pairs = step_3_qa_pairs + step_6_qa_pairs + step_9_qa_pairs

    # with open(file_path, "w") as f:
    #     for qa_pair in all_qa_pairs:
    #         f.write(json.dumps(qa_pair.to_dict()) + "\n")

    # ablation step 2: validation
    ## ablation step 2.1: validate aperture 30
    aperture_30_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/aperture_30_ordering_samples.jsonl"
    # aperture_30_step_3_generator = MultiStepForwardDynamicsAblationGenerator(step_length=3)
    # aperture_30_step_3_qa_dict_list = aperture_30_step_3_generator.get_valid_qa_pairs(file_path, task_data)
    # aperture_30_step_6_generator = MultiStepForwardDynamicsAblationGenerator(step_length=6)
    # aperture_30_step_6_qa_dict_list = aperture_30_step_6_generator.get_valid_qa_pairs(file_path, task_data)
    # aperture_30_step_9_generator = MultiStepForwardDynamicsAblationGenerator(step_length=9)
    # aperture_30_step_9_qa_dict_list = aperture_30_step_9_generator.get_valid_qa_pairs(file_path, task_data)

    # aperture_30_qa_dict_list = aperture_30_step_3_qa_dict_list + aperture_30_step_6_qa_dict_list + aperture_30_step_9_qa_dict_list

    # with open(aperture_30_file_path, "w") as f:
    #     for qa_dict in aperture_30_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")

    ## ablation step 2.2: validate high camera
    high_camera_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/high_camera_ordering_samples.jsonl"
    # high_camera_step_3_generator = MultiStepForwardDynamicsAblationGenerator(step_length=3)
    # high_camera_step_3_qa_dict_list = high_camera_step_3_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)
    # high_camera_step_6_generator = MultiStepForwardDynamicsAblationGenerator(step_length=6)
    # high_camera_step_6_qa_dict_list = high_camera_step_6_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)
    # high_camera_step_9_generator = MultiStepForwardDynamicsAblationGenerator(step_length=9)
    # high_camera_step_9_qa_dict_list = high_camera_step_9_generator.get_valid_qa_pairs(aperture_30_file_path, task_data)

    # high_camera_qa_dict_list = high_camera_step_3_qa_dict_list + high_camera_step_6_qa_dict_list + high_camera_step_9_qa_dict_list

    # with open(high_camera_file_path, "w") as f:
    #     for qa_dict in high_camera_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")

    ## ablation step 2.3: validate low camera
    low_camera_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/low_camera_ordering_samples.jsonl"
    # low_camera_step_3_generator = MultiStepForwardDynamicsAblationGenerator(step_length=3)
    # low_camera_step_3_qa_dict_list = low_camera_step_3_generator.get_valid_qa_pairs(high_camera_file_path, task_data)
    # low_camera_step_6_generator = MultiStepForwardDynamicsAblationGenerator(step_length=6)
    # low_camera_step_6_qa_dict_list = low_camera_step_6_generator.get_valid_qa_pairs(high_camera_file_path, task_data)
    # low_camera_step_9_generator = MultiStepForwardDynamicsAblationGenerator(step_length=9)
    # low_camera_step_9_qa_dict_list = low_camera_step_9_generator.get_valid_qa_pairs(high_camera_file_path, task_data)

    # low_camera_qa_dict_list = low_camera_step_3_qa_dict_list + low_camera_step_6_qa_dict_list + low_camera_step_9_qa_dict_list

    # with open(low_camera_file_path, "w") as f:
    #     for qa_dict in low_camera_qa_dict_list:
    #         f.write(json.dumps(qa_dict) + "\n")


    # for step 3, 6, 9, each randomly sample 50 dicts from the file
    foward_sampled_file_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/foward_ablation_ordering_samples.jsonl"
    # foward_dict = {
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
    #             foward_dict['3'].append(current_data)
    #         elif '6_steps' in id:
    #             foward_dict['6'].append(current_data)
    #         elif '9_steps' in id:
    #             foward_dict['9'].append(current_data)
    # ## 2. sample 50 dicts from the foward_dict
    # for key in foward_dict.keys():
    #     foward_dict[key] = random.sample(foward_dict[key], 50)
    #     sampled_list.extend(foward_dict[key])
    # ## 3. write the sampled list to the file
    # with open(foward_sampled_file_path, "w") as f:
    #     for qa_dict in sampled_list:
    #         f.write(json.dumps(qa_dict) + "\n")


    # generate mother QA pairs
    mother_qa_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_raw_files/segmented_ablation/forward_mother_ordering_samples.jsonl"
    mother_qa_generator = MultiStepForwardDynamicsAblationGenerator(step_length=3)
    mother_qa_pairs = mother_qa_generator.generate_mother_QA_pairs(task_data, foward_sampled_file_path)
    with open(mother_qa_path, "w") as f:
        for qa_pair in mother_qa_pairs:
            f.write(json.dumps(qa_pair.to_dict()) + "\n")

    
    
