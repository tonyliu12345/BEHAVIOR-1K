import os
import base64
import json
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import logging
from typing import List, Dict, Any, Optional, Set

# Add debug mode flag early
DEBUG_MODE = False  # Set to True for testing without API calls
DRY_RUN_MODE = False  # Set to True for testing without API calls (same as DEBUG_MODE but more explicit)

# Conditional imports
if not DEBUG_MODE and not DRY_RUN_MODE:
    from openai import OpenAI
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
file_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA_ablation/behavior_eqa_ordering_ablation.jsonl"
output_folder = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/ablation_model_outputs"
data_root_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset"  # Root path for resolving relative image paths
model_id = "gpt-5-mini-2025-08-07"
output_path = os.path.join(output_folder, f"behavior_eqa_ordering_ablation_{model_id}.jsonl")
NUM_PROCESSES = 16 if not DRY_RUN_MODE else 2  # Use fewer processes in dry run mode

def open_image_with_exif(full_path):
    img = Image.open(full_path)
    try:
        # Use more standard PIL approach with getexif()
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
    json_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".jsonl"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_images(image_paths):
    images = []
    for path in image_paths:
        if os.path.exists(path):
            image = open_image_with_exif(path)

            ###RGBA
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            images.append(image)
        else:
            print(f"Warning: Image path {path} does not exist.")
    return images

# Function to encode the image
def encode_image(image_path):
  try:
    # Resolve relative path to absolute path
    absolute_path = resolve_image_path(image_path)
    # Open image with proper orientation handling
    img = open_image_with_exif(absolute_path)
    # Convert to RGB if needed
    if img.mode == 'RGBA':
      img = img.convert('RGB')
    # Save to a bytes buffer and encode
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
  except Exception as e:
    print(f"Error encoding image {image_path}: {e}")
    # Fallback to direct file reading if there's an error
    absolute_path = resolve_image_path(image_path)
    with open(absolute_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

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
        # try to open the image
        try:
            Image.open(absolute_path)
        except Exception as e:
            print(f"Error in opening image {absolute_path}: {e}")
            return False
    return True

def get_openai_response(model, prompt, image_paths):
    try:
        # In dry run mode, return a mock response
        if DRY_RUN_MODE:
            logger.info(f"DRY RUN: Would process {len(image_paths)} images with model {model}")
            logger.info(f"DRY RUN: Prompt length: {len(prompt)} characters")
            # Return a mock response that matches the expected format for ordering tasks
            import random
            num_steps = len(image_paths) - 1  # Subtract 1 for current state image
            if num_steps > 0:
                mock_order = list(range(1, num_steps + 1))  # [1, 2, 3, ...] format
                random.shuffle(mock_order)  # Shuffle to simulate model response
                return f"[{', '.join(map(str, mock_order))}]"
            return "[1]"
        
        # Create a new client for each process to avoid sharing issues
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        base64_images = [encode_image(image) for image in image_paths]
        user_content = [{"type": "text", "text": prompt}]
        base64_images = [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                }
                                for base64_image in base64_images
                            ]
        user_content.extend(base64_images)
        messages = [{"role": "user", "content": user_content}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        model_answer = completion.choices[0].message.content
        return model_answer
    except Exception as e:
        logger.error(f"Error in get_openai_response: {e}")
        return None

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
        # If file exists but can't be read, backup and start fresh
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
        
        # In the new format, all images are in the 'images' array
        # This includes current state + all future states for both forward and inverse dynamics
        all_image_paths = item_data.get("images", [])
        
        if not all_image_paths:
            logger.error(f"No images found for {q_id}")
            return None
        
        # Validate image paths
        if not test_image_path_readable(all_image_paths):
            logger.error(f"Image paths not readable for {q_id}")
            return None
        
        # Get OpenAI response
        answer = get_openai_response(model_id, question, all_image_paths)
        time.sleep(1)
        
        if answer is None:
            logger.error(f"Failed to get response for {q_id}")
            return None
        
        # Prepare result
        result_data = item_data.copy()
        result_data["answer"] = answer
        result_data["processed_at"] = time.time()
        
        logger.info(f"Successfully processed {q_id} ({data_type})")
        if not DRY_RUN_MODE:
            time.sleep(1)  # Only sleep in real API mode
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing item {item_data.get('id', 'unknown')}: {e}")
        return None

def write_result_safely(result: Dict[str, Any], output_path: str):
    """Safely write a single result to the output file."""
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  # Ensure data is written immediately
    except Exception as e:
        logger.error(f"Error writing result for {result.get('id', 'unknown')}: {e}")

def main():
    # Log current mode
    if DRY_RUN_MODE:
        logger.info("=" * 50)
        logger.info("RUNNING IN DRY RUN MODE - NO API CALLS WILL BE MADE")
        logger.info("=" * 50)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Check existing results
    completed_ids = check_existing_results(output_path)
    
    # Load all testing data
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
    
    # Filter out already completed items
    pending_data = [item for item in testing_data if item.get("id", "") not in completed_ids]
    
    logger.info(f"Total items: {len(testing_data)}")
    logger.info(f"Already completed: {len(completed_ids)}")
    logger.info(f"Pending items: {len(pending_data)}")
    
    # Log data type summary for pending items
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
    
    # Process items with multiprocessing
    logger.info(f"Starting processing with {NUM_PROCESSES} processes")
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # Submit all pending tasks
        future_to_item = {executor.submit(process_single_item, item): item for item in pending_data}
        
        completed_count = 0
        failed_count = 0
        
        # Process completed tasks as they finish
        for future in tqdm(as_completed(future_to_item), total=len(pending_data), desc="Processing"):
            item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    # Write result immediately when it's ready
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