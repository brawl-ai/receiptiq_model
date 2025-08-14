import shutil
import os
import glob
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5-mini"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

IMAGE_BASE_URL = "https://receiptiq-model-finetuning-receipts.t3.storageapi.dev"

SYSTEM_PROMPT = """
Extract receipt data as valid JSON only.

Rules:
- Each leaf node: {"value": string|int|float, "bounding_box": [x,y,w,h], "description": short string}.
- snake_case keys.
- Arrays only for repeated objects of identical structure (e.g., items list).
- Give best estimate if unsure; no apologies, no commentary.
- Output ONLY a single JSON object from { to } with no code fences or text outside.

Example:
{"store_name":{"value":"SuperMart","bounding_box":[12,45,250,32],"description":"Name of the store"},"date":{"value":"2025-08-01","bounding_box":[300,45,180,28],"description":"Purchase date"},"items":[{"item_name":{"value":"Apples","bounding_box":[50,120,200,25],"description":"Name of purchased item"},"quantity":{"value":2,"bounding_box":[270,120,50,25],"description":"Number of units purchased"},"unit_price":{"value":1.5,"bounding_box":[340,120,60,25],"description":"Price per unit"}}],"total_amount":{"value":3.0,"bounding_box":[340,500,80,28],"description":"Total purchase amount"}}
"""

client = OpenAI(api_key=API_KEY)

def get_image_files(input_dir: str):
    return glob.glob(os.path.join(input_dir, "*.jpg"))

def prepare_batch_files(image_files_list: List[str], output_dir: str, max_items_per_batch: int):
    IMAGE_BASE_URL = "https://receiptiq-model-finetuning-receipts.t3.storageapi.dev"
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)
    batch_id = 0
    current_batch = []
    batch_files = []
    for img_file in image_files_list:
        file_name = img_file.split("/")[-1]
        img_url = f"{IMAGE_BASE_URL}/{file_name}"
        payload = {
            "custom_id": img_file,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user","content": [{"type": "image_url","image_url": {"url": img_url}}]}
                ]
            }
        }
        current_batch.append(payload)
        if len(current_batch) >= max_items_per_batch:
            batch_path = os.path.join(output_dir, f"v2_batch_{batch_id}.jsonl")
            with open(batch_path, "w") as f:
                for item in current_batch:
                    f.write(json.dumps(item) + "\n")
            batch_id += 1
            current_batch = []
            batch_files.append(batch_path)
    if current_batch:
        batch_path = os.path.join(output_dir, f"v2_batch_{batch_id}.jsonl")
        batch_files.append(batch_path)
        with open(batch_path, "w") as f:
            for item in current_batch:
                f.write(json.dumps(item) + "\n")
    return batch_files

if __name__ == "__main__":
    images_dir = "datasets/images2"
    batches_dir = "datasets/batches2"
    completion_window = "24h"
    max_items_per_batch = 500
    images_list = get_image_files(images_dir)
    batch_files = prepare_batch_files(image_files_list=images_list,output_dir=batches_dir, max_items_per_batch=max_items_per_batch)
    for idx, batch_file in enumerate(batch_files):
        with open(batch_file, "rb") as f:
            file_id = client.files.create(file=f, purpose="batch").id
            print("file created ",file_id)