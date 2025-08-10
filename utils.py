import os
import gc
from typing import Dict
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset as TorchDataset

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def download_and_save_model(local_model_path: str, model_id: str):
    """Download and save the complete multimodal model."""
    print("Downloading complete multimodal model...")

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Saving to {local_model_path}...")
    os.makedirs(local_model_path, exist_ok=True)
    model.save_pretrained(local_model_path)
    processor.save_pretrained(local_model_path)

    # 🔥 Drop from memory to avoid GPU crowding
    del model, processor
    clear_cuda()

    print("Model saved successfully!")

def load_model_quantized(local_model_path: str, model_id: str):
    """Load the model with quantization for training."""
    if not os.path.exists(local_model_path):
        download_and_save_model(local_model_path, model_id)
    else:
        clear_cuda()  # in case something else is lurking

    print("Loading quantized model from local path...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )

    processor = AutoProcessor.from_pretrained(
        local_model_path,
        local_files_only=True
    )

    print("Quantized model loaded successfully!")
    return model, processor

class ReceiptIQModelDataLoader(TorchDataset):
    def __init__(self, dataset, max_len: int, tokenizer):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        receipt = self.dataset[idx]
        prompt_input_ids = torch.tensor(receipt["input_ids"])
        prompt_attention_mask = torch.tensor(receipt["attention_mask"])
        output_input_ids = torch.tensor(receipt["output_ids"])
        output_attention_mask = torch.tensor(receipt["output_attention_mask"])
        
        combined_input_ids = torch.cat((prompt_input_ids, output_input_ids))
        combined_attention_mask = torch.cat((prompt_attention_mask, output_attention_mask))
        labels = combined_input_ids.clone()
        prompt_length = prompt_input_ids.shape[0]
        labels[:prompt_length] = -100

        # Pad to self.max_len
        current_len = combined_input_ids.shape[0]
        if current_len < self.max_len:
            padding_len = self.max_len - current_len
            combined_input_ids = torch.cat([combined_input_ids, torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            combined_attention_mask = torch.cat([combined_attention_mask, torch.zeros(padding_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_len,), -100, dtype=torch.long)])

        return {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "labels": labels,
            "pixel_values": torch.tensor(receipt["pixel_values"]),
            "aspect_ratio_ids": torch.tensor(receipt["aspect_ratio_ids"]),
            "aspect_ratio_mask": torch.tensor(receipt['aspect_ratio_mask']),
        }