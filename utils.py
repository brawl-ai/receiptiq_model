import os
import gc
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

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