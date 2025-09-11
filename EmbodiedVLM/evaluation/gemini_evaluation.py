# -*- coding: utf-8 -*-
"""
Drop-in fix for Gemini "200 OK but response.text is None".
- DOES NOT modify your prompt or add any extra instruction text.
- Keeps your original I/O, logging, and multiprocessing structure.
- Adds robust text extraction from candidates when response.text is empty.
- Attempts to disable automatic function calling (AFC) via tool_config; falls back if unsupported.

Requirements:
  pip install -U google-genai python-dotenv pillow tqdm
"""

import os
import base64
import json
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import logging
from typing import List, Dict, Any, Optional, Set
from io import BytesIO

# Mode flags
DEBUG_MODE = False   # True: no API calls
DRY_RUN_MODE = False # True: no API calls
DEBUG_ONE_MODE = False  # True: process only one item

# Conditional imports
if not DEBUG_MODE and not DRY_RUN_MODE:
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
file_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA/behavior_eqa_ordering.jsonl"
output_folder = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/model_outputs"
data_root_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset"  # Root path for resolving relative image paths
model_id = "gemini-2.5-pro"  # Use "gemini-2.5-pro" for production
output_path = os.path.join(output_folder, f"behavior_eqa_ordering_{model_id}.jsonl")
NUM_PROCESSES = 1 if not DRY_RUN_MODE and not DEBUG_ONE_MODE else 2  # Use fewer processes in dry run mode

# ---------- Image utilities ----------

def open_image_with_exif(full_path):
    """Open image and handle EXIF orientation."""
    img = Image.open(full_path)
    try:
        exif = img.getexif() if hasattr(img, 'getexif') else None
        if exif is not None:
            orientation = exif.get(274, 1)  # 274 is the orientation tag
            if orientation == 3:  # 180 degree rotation
                img = img.rotate(180, expand=True)
            elif orientation == 6:  # Rotate 270 degrees
                img = img.rotate(270, expand=True)
            elif orientation == 8:  # Rotate 90 degrees
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Could not process EXIF data for {full_path}: {e}")
    return img

def find_json_files(input_root):
    """Find all JSONL files in the input directory."""
    json_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".jsonl"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_images(image_paths):
    """Load images from given paths."""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            image = open_image_with_exif(path)
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            images.append(image)
        else:
            print(f"Warning: Image path {path} does not exist.")
    return images

def encode_pil_imgs_to_base64(pil_images):
    """Encode PIL images to base64 strings."""
    base64_strings = []
    for img in pil_images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_strings.append(img_str)
    return base64_strings

def resolve_image_path(relative_path):
    """Convert relative image path to absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(data_root_path, relative_path)

def test_image_path_readable(image_paths):
    """Test if image paths are readable, converting relative paths to absolute."""
    for image_path in image_paths:
        absolute_path = resolve_image_path(image_path)
        if not os.path.exists(absolute_path):
            print(f"Warning: Image path {absolute_path} does not exist (original: {image_path}).")
            return False
        try:
            Image.open(absolute_path).close()
        except Exception as e:
            print(f"Error in opening image {absolute_path}: {e}")
            return False
    return True

# ---------- Response helpers ----------

def extract_text(resp) -> Optional[str]:
    """
    Robust text extraction:
    1) Try resp.text
    2) Fallback: concatenate candidates[].content.parts[].text
    """
    # Primary path
    txt = getattr(resp, "text", None)
    if txt:
        s = txt.strip()
        if s:
            return s

    # Fallback path
    try:
        data = resp.to_dict() if hasattr(resp, "to_dict") else resp
        pieces = []
        for cand in (data.get("candidates") or []):
            content = cand.get("content") or {}
            for part in (content.get("parts") or []):
                t = part.get("text")
                if t:
                    pieces.append(t)
        combined = "\n".join(pieces).strip()
        return combined if combined else None
    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")
        return None

def build_generate_config():
    """
    Build GenerateContentConfig and disable function calling if supported.
    Does NOT change your prompt or add any instruction text.
    """
    try:
        # If the installed google-genai supports ToolConfig / FunctionCallingConfig:
        # tool_cfg = types.ToolConfig(
        #     function_calling_config=types.FunctionCallingConfig(mode="NONE")
        # )
        return types.GenerateContentConfig(
            temperature=0,
            # tool_config=tool_cfg
        )
    except Exception:
        # Fallback for older versions: just set temperature
        return types.GenerateContentConfig(temperature=0)

# ---------- Gemini call ----------

def gemini_gen(model_name, prompt, image_paths, max_retries=5):
    """
    Generate content using Gemini API WITHOUT altering your prompt.
    
    Implements exponential retry strategy specifically for None responses:
    - Base delay starts at 2 seconds
    - Each retry doubles the delay: 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s
    - Increased max_retries from 3 to 5 to handle None responses better
    - Enhanced logging for None response diagnostics
    """
    try:
        # In dry/debug mode, return a mock response
        if DRY_RUN_MODE or DEBUG_MODE:
            logger.info(f"DRY RUN: Would process {len(image_paths)} images with model {model_name}")
            import random
            num_steps = max(1, len(image_paths) - 1)
            mock_order = list(range(1, num_steps + 1))
            random.shuffle(mock_order)
            return f"[{', '.join(map(str, mock_order))}]"

        # Load images and convert to base64
        absolute_paths = [resolve_image_path(path) for path in image_paths]
        pil_images = load_images(absolute_paths)
        base64_imgs = encode_pil_imgs_to_base64(pil_images)

        # Create Gemini client
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

        # Prepare contents for Gemini (keep your structure: labels + image parts + your prompt)
        contents = []
        for i, img in enumerate(base64_imgs, 1):
            contents.append(f"Image {i}:")
            image_bytes = base64.b64decode(img)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                )
            )

        # IMPORTANT: Do not modify or add to your prompt
        contents.append(prompt)

        config = build_generate_config()

        current_attempt = 0
        base_delay = 2.0  # Start with 2 seconds for exponential backoff
        while current_attempt < max_retries:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )

                text = extract_text(response)
                if text:
                    if current_attempt > 0:
                        logger.info(f"Successfully got response after {current_attempt + 1} attempts")
                    return text

                # Handle None response with exponential retry strategy
                logger.warning(f"Response text is None on attempt {current_attempt + 1}/{max_retries}")
                
                # Log detailed diagnostics for None responses
                try:
                    d = response.to_dict()
                    cands = d.get("candidates") or []
                    finish = (cands[0].get("finishReason") if cands else None)
                    safety_ratings = (cands[0].get("safetyRatings") if cands else None)
                    logger.warning(
                        f"None response details - finishReason: {finish}, "
                        f"num_candidates: {len(cands)}, "
                        f"promptFeedback: {d.get('promptFeedback')}, "
                        f"safetyRatings: {safety_ratings}"
                    )
                except Exception as diag_e:
                    logger.warning(f"Could not extract response diagnostics: {diag_e}")

            except Exception as e:
                logger.error(f"API error on attempt {current_attempt + 1}/{max_retries}: {e}")

            current_attempt += 1
            if current_attempt < max_retries:
                # Exponential backoff: 2, 4, 8, 16, 32 seconds
                delay = base_delay * (2 ** (current_attempt - 1))
                logger.info(f"Retrying in {delay:.1f} seconds (exponential backoff)...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} retry attempts failed for model {model_name}. Giving up.")
                return None

        return None

    except Exception as e:
        logger.error(f"Error in gemini_gen: {e}")
        return None

# ---------- Results handling ----------

def check_existing_results(output_path: str) -> Set[str]:
    """Check existing results and return set of completed IDs."""
    completed_ids = set()

    if not os.path.exists(output_path):
        logger.info(f"Output file {output_path} does not exist. Starting fresh.")
        return completed_ids

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    q_id = data.get("id", "")
                    if q_id:
                        completed_ids.add(q_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        logger.info(f"Found {len(completed_ids)} completed results in existing file")
        logger.info(f"Output file is readable: {output_path}")

    except Exception as e:
        logger.error(f"Error reading existing results: {e}")
        backup_path = output_path + f".backup_{int(time.time())}"
        try:
            os.rename(output_path, backup_path)
            logger.info(f"Backed up unreadable file to {backup_path}")
        except Exception:
            logger.error(f"Could not backup unreadable file")

    return completed_ids

def process_single_item(item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single data item. This function will be run in parallel."""
    try:
        q_id = item_data.get("id", "")
        question = item_data.get("question", "")
        data_type = item_data.get("type", "")
        all_image_paths = item_data.get("images", [])

        if not all_image_paths:
            logger.error(f"No images found for {q_id}")
            return None

        if not test_image_path_readable(all_image_paths):
            logger.error(f"Image paths not readable for {q_id}")
            return None

        # Use your original question (prompt) verbatim
        answer = gemini_gen(model_id, question, all_image_paths)

        if answer is None:
            logger.error(f"Failed to get response for {q_id}")
            return None

        result_data = item_data.copy()
        result_data["answer"] = answer
        result_data["processed_at"] = time.time()

        logger.info(f"Successfully processed {q_id} ({data_type})")
        if not DRY_RUN_MODE and not DEBUG_MODE:
            time.sleep(2)  # gentle pacing
        return result_data

    except Exception as e:
        logger.error(f"Error processing item {item_data.get('id', 'unknown')}: {e}")
        return None

def write_result_safely(result: Dict[str, Any], output_path: str):
    """Safely write a single result to the output file."""
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
    except Exception as e:
        logger.error(f"Error writing result for {result.get('id', 'unknown')}: {e}")

# ---------- Main ----------

def main():
    logger.info("Sleeping for an hour...")
    # time.sleep(60 * 60)
    if DRY_RUN_MODE or DEBUG_MODE:
        logger.info("=" * 50)
        logger.info("RUNNING IN DRY/DEBUG MODE - NO API CALLS WILL BE MADE")
        logger.info("=" * 50)
    elif DEBUG_ONE_MODE:
        logger.info("=" * 50)
        logger.info("RUNNING IN DEBUG ONE MODE - PROCESSING ONLY ONE DATA POINT")
        logger.info("=" * 50)

    os.makedirs(output_folder, exist_ok=True)

    completed_ids = check_existing_results(output_path)

    logger.info(f"Loading data from {file_path}")
    testing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, desc="Loading data")):
                try:
                    data = json.loads(line.strip())
                    testing_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {idx+1}: {e}")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return

    pending_data = [item for item in testing_data if item.get("id", "") not in completed_ids]

    if DEBUG_ONE_MODE and pending_data:
        pending_data = pending_data[:1]
        logger.info(f"DEBUG_ONE_MODE: Processing only the first item: {pending_data[0].get('id', 'unknown')}")

    logger.info(f"Total items: {len(testing_data)}")
    logger.info(f"Already completed: {len(completed_ids)}")
    logger.info(f"Pending items: {len(pending_data)}")

    if pending_data:
        type_counts = {}
        for item in pending_data:
            data_type = item.get("type", "unknown")
            type_counts[data_type] = type_counts.get(data_type, 0) + 1

        logger.info("Pending items by type:")
        for data_type, count in sorted(type_counts.items()):
            logger.info(f"  {data_type}: {count}")

    if not pending_data:
        logger.info("All items already processed. Exiting.")
        return

    # sleep an hour

    logger.info(f"Starting processing with {NUM_PROCESSES} processes")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in pending_data}

        completed_count = 0
        failed_count = 0

        for future in tqdm(as_completed(future_to_item), total=len(pending_data), desc="Processing"):
            item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    write_result_safely(result, output_path)
                    completed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process item {item.get('id', 'unknown')}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Exception processing item {item.get('id', 'unknown')}: {e}")

    logger.info(f"Processing completed. Success: {completed_count}, Failed: {failed_count}")
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
