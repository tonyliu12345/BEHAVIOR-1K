# -*- coding: utf-8 -*-
"""
InternVL3.5-241B-A28B 推理脚本（兼容你的 OpenAI 评测框架）
- 依赖：pip install openai pillow python-dotenv tqdm
- 令牌：在环境变量或 .env 中设置 INTERNLM_API_KEY=<你的token>
- 端点：base_url=https://chat.intern-ai.org.cn/api/v1/
文档依据（多模态消息结构、SDK 兼容、模型名/限流说明）：
https://internlm.intern-ai.org.cn/doc/docs/Chat/
"""

import os
import base64
import json
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
from typing import List, Dict, Any, Optional, Set
import re
import random

# ===== 可配置：调试/空跑 =====
DEBUG_MODE = False     # True: 不发起真实 API（但仍走流程）
DRY_RUN_MODE = False   # True: 与上类似，更显式

# 条件导入
if not DEBUG_MODE and not DRY_RUN_MODE:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

# ===== 日志 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("internvl_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ===== 路径与模型配置（根据你的环境改） =====
input_idx = 2
file_path = f"/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA_ablation/behavior_eqa_ordering_ablation.jsonl"
output_folder = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/evaluation/ablation_model_outputs"
data_root_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset"

# 目标模型：InternVL3.5 241B
model_id = "internvl3.5-241b-a28b"

# 输出文件
output_path = os.path.join(output_folder, f"behavior_eqa_ordering_ablation_{model_id}.jsonl")

# 并发进程数（注意 QPM=30 的限流；过高并发可能频繁 429）
NUM_PROCESSES = 5 if not DRY_RUN_MODE else 2

# ===== InternLM OpenAI 兼容 SDK 的连接配置 =====
INTERNLM_API_BASE = os.getenv("INTERNLM_API_BASE", "https://chat.intern-ai.org.cn/api/v1/")
INTERNLM_API_KEY = os.getenv("INTERNLM_API_KEY")

# ===== 图像预处理 =====
def open_image_with_exif(full_path):
    img = Image.open(full_path)
    try:
        exif = img.getexif() if hasattr(img, "getexif") else None
        if exif is not None:
            orientation = exif.get(274, 1)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Could not process EXIF data for {full_path}: {e}")
    return img

def resolve_image_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(data_root_path, relative_path)

def test_image_path_readable(image_paths):
    for image_path in image_paths:
        absolute_path = resolve_image_path(image_path)
        if not os.path.exists(absolute_path):
            print(f"Warning: Image path {absolute_path} does not exist (original: {image_path}).")
            return False
        try:
            Image.open(absolute_path)
        except Exception as e:
            print(f"Error in opening image {absolute_path}: {e}")
            return False
    return True

def encode_image(image_path):
    try:
        absolute_path = resolve_image_path(image_path)
        img = open_image_with_exif(absolute_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        absolute_path = resolve_image_path(image_path)
        with open(absolute_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

# ===== 结果断点续跑 =====
def check_existing_results(output_path: str) -> Set[str]:
    completed_ids = set()
    if not os.path.exists(output_path):
        logger.info(f"Output file {output_path} does not exist. Starting fresh.")
        return completed_ids

    try:
        with open(output_path, "r", encoding="utf-8") as f:
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

# ===== 指数退避请求 =====
def send_with_backoff(client, model, messages, max_attempts=6, max_wait=32):
    """
    指数退避：1,2,4,8,16,32 秒（含随机抖动最多 +0.5s）
    返回 message.content 文本或 None
    """
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0
            )
            if completion and completion.choices:
                content = getattr(completion.choices[0].message, "content", None)
                if content:
                    return content
            raise RuntimeError("Empty content or no choices in response")
        except Exception as e:
            wait = min(2 ** (attempt - 1), max_wait) + random.uniform(0, 0.5)
            logger.warning(f"[Backoff] attempt {attempt}/{max_attempts} failed: {e}; sleep {wait:.1f}s")
            time.sleep(wait)
    return None

# ===== 解析排序答案（解析失败 => 失败） =====
def parse_ordering_answer(answer_text: str, num_steps: int) -> Optional[List[int]]:
    """
    目标：从答案文本中解析出一个包含 1..num_steps 的不重复序列。
    - 优先找形如 [1, 3, 2] 的列表
    - 否则提取所有数字，按出现顺序去重，截取前 num_steps 个
    - 长度必须==num_steps，且都在 [1..num_steps]
    """
    if num_steps <= 0:
        return [1]

    if not answer_text:
        return None

    # 尝试 [ ... ] 内的数字
    m = re.search(r"\[([^\]]+)\]", answer_text)
    cand_nums: List[int] = []
    if m:
        cand_nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    else:
        cand_nums = [int(x) for x in re.findall(r"\d+", answer_text)]

    if not cand_nums:
        return None

    # 去重 + 筛范围 + 截取
    seq = []
    seen = set()
    for n in cand_nums:
        if 1 <= n <= num_steps and n not in seen:
            seq.append(n)
            seen.add(n)
        if len(seq) == num_steps:
            break

    if len(seq) != num_steps:
        return None
    return seq

# ===== InternVL 请求 =====
def get_internvl_response(model: str, prompt: str, image_paths: List[str]) -> Optional[str]:
    try:
        if DRY_RUN_MODE:
            logger.info(f"DRY RUN: Would process {len(image_paths)} images with model {model}")
            logger.info(f"DRY RUN: Prompt length: {len(prompt)} characters")
            # 模拟一个合法格式的随机顺序
            import random as _rand
            num_steps = max(1, len(image_paths) - 1)
            order = list(range(1, num_steps + 1))
            _rand.shuffle(order)
            return f"[{', '.join(map(str, order))}]"

        # 初始化客户端（每进程/每次调用单独实例，避免句柄共享）
        if not INTERNLM_API_KEY:
            raise RuntimeError("INTERNLM_API_KEY 未设置。请在环境变量或 .env 中配置。")
        client = OpenAI(api_key=INTERNLM_API_KEY, base_url=INTERNLM_API_BASE)

        # 按文档构造多模态消息：user.content 为数组（text + 多个 image_url）
        # 参考文档的多模态示例。支持 URL 或 data:base64。此处采用 base64 内联。:contentReference[oaicite:5]{index=5}
        user_content = [{"type": "text", "text": prompt}]

        base64_images = [encode_image(p) for p in image_paths]
        user_content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                }
                for b64 in base64_images
            ]
        )

        messages = [{"role": "user", "content": user_content}]

        # 指数退避发送
        answer_text = send_with_backoff(client, model, messages, max_attempts=6, max_wait=32)
        return answer_text
    except Exception as e:
        logger.error(f"Error in get_internvl_response: {e}")
        return None

# ===== 单条样本处理（含解析校验） =====
def process_single_item(item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

        # 请求 InternVL
        raw_answer = get_internvl_response(model_id, question, all_image_paths)
        time.sleep(0.5)  # 轻微节流，降低 429 概率

        if raw_answer is None:
            logger.error(f"Failed to get response for {q_id}")
            return None

        # 解析排序答案（解析失败 => 失败）
        num_steps = max(1, len(all_image_paths) - 1)
        parsed = parse_ordering_answer(raw_answer, num_steps)
        if parsed is None:
            logger.error(f"Parsing failed for {q_id}. raw_answer={raw_answer!r}")
            return None

        # 写出时 answer 保持简洁数组字符串，另存 raw_answer 便于排查
        result_data = item_data.copy()
        result_data["answer"] = raw_answer
        result_data["parsed_answer"] = f"[{', '.join(map(str, parsed))}]"
        result_data["processed_at"] = time.time()

        logger.info(f"Successfully processed {q_id} ({data_type})")
        return result_data

    except Exception as e:
        logger.error(f"Error processing item {item_data.get('id', 'unknown')}: {e}")
        return None

# ===== 可靠写文件 =====
def write_result_safely(result: Dict[str, Any], output_path: str):
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        logger.error(f"Error writing result for {result.get('id', 'unknown')}: {e}")

# ===== 主流程 =====
def main():
    # 模式提示
    if DRY_RUN_MODE:
        logger.info("=" * 50)
        logger.info("RUNNING IN DRY RUN MODE - NO API CALLS WILL BE MADE")
        logger.info("=" * 50)

    os.makedirs(output_folder, exist_ok=True)

    completed_ids = check_existing_results(output_path)

    # 读入数据
    logger.info(f"Loading data from {file_path}")
    testing_data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
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

    # 并行处理
    logger.info(f"Starting processing with {NUM_PROCESSES} processes")
    completed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in pending_data}

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
