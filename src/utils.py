"""
Utility functions for hardware detection and Vietnamese text processing
"""

import torch
import re
import unicodedata
from typing import Dict, Tuple
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def auto_detect_hardware() -> Tuple[str, Dict]:
    """
    Tự động phát hiện phần cứng và trả về cấu hình phù hợp
    
    Returns:
        mode (str): 'server_mode' hoặc 'laptop_mode'
        info (dict): Thông tin chi tiết về hardware
    """
    info = {
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": 0,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    # Check CUDA availability
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Get VRAM (GPU memory)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024**3), 2)
        
        logger.info(f"🎮 GPU Detected: {info['gpu_name']}")
        logger.info(f"💾 VRAM: {info['vram_gb']} GB")
        
        # Quyết định mode dựa trên VRAM
        if info["vram_gb"] >= 10:
            mode = "server_mode"
            logger.info("✅ Running in SERVER MODE (High Performance)")
        else:
            mode = "laptop_mode"
            logger.info("⚠️  Running in LAPTOP MODE (Memory Optimized)")
            if info["vram_gb"] < 5:
                logger.warning("🔴 VRAM < 5GB: 8-bit loading is MANDATORY!")
    else:
        mode = "laptop_mode"
        logger.info("💻 CPU Detected: Running in LAPTOP MODE")
        logger.info(f"💾 RAM: {info['ram_gb']} GB")
    
    return mode, info


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    1. Chuẩn hóa Unicode (NFD -> NFC)
    2. Chuẩn hóa vị trí dấu thanh (tone normalization)
    3. Loại bỏ khoảng trắng thừa
    4. Lowercase
    
    Args:
        text (str): Văn bản gốc
    
    Returns:
        str: Văn bản đã chuẩn hóa
    """
    if not text:
        return ""
    
    # 1. Chuẩn hóa Unicode về dạng NFC (Composed form)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Chuẩn hóa vị trí dấu thanh (Vietnamese tone marks)
    # Ví dụ: "hoà" -> "hòa", "huỷ" -> "hủy"
    text = normalize_vietnamese_tone(text)
    
    # 3. Lowercase
    text = text.lower()
    
    # 4. Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Loại bỏ ký tự đặc biệt không mong muốn (giữ lại dấu câu cơ bản)
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ,.]', '', text)
    
    return text


def normalize_vietnamese_tone(text: str) -> str:
    """
    Chuẩn hóa vị trí dấu thanh tiếng Việt về dạng chuẩn
    
    Quy tắc:
    - Với "oa", "oe", "uy": dấu đặt ở nguyên âm thứ 2 (oà, oè, uỳ)
    - Với các trường hợp khác: giữ nguyên
    
    Args:
        text (str): Văn bản gốc
    
    Returns:
        str: Văn bản với dấu thanh đã được chuẩn hóa
    """
    # Bảng ánh xạ: ký tự sai vị trí dấu -> ký tự đúng vị trí dấu
    tone_map = {
        # oa: dấu nên ở 'a' chứ không phải 'o'
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
        # oe: dấu nên ở 'e' chứ không phải 'o'  
        'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
        # uy: dấu nên ở 'y' chứ không phải 'u'
        'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
    }
    
    for wrong, correct in tone_map.items():
        text = text.replace(wrong, correct)
    
    return text


def calculate_wer(predictions: list, references: list) -> float:
    """
    Tính Word Error Rate (WER) - metric chính cho ASR
    
    WER = (S + D + I) / N
    S: Substitutions (từ thay thế)
    D: Deletions (từ bị xóa)
    I: Insertions (từ thêm vào)
    N: Tổng số từ trong reference
    
    Args:
        predictions (list): Danh sách văn bản dự đoán
        references (list): Danh sách văn bản tham chiếu
    
    Returns:
        float: WER score (càng thấp càng tốt)
    """
    import jiwer
    
    # Normalize trước khi tính
    predictions = [normalize_vietnamese_text(p) for p in predictions]
    references = [normalize_vietnamese_text(r) for r in references]
    
    wer = jiwer.wer(references, predictions)
    return wer


def setup_logging(output_dir: str):
    """
    Thiết lập logging cho training
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def print_model_size(model):
    """
    In kích thước model và số lượng parameters
    """
    param_size = 0
    trainable_params = 0
    all_param = 0
    
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    param_size = all_param * 4 / (1024 ** 2)  # Assuming float32, convert to MB
    
    logger.info(f"📊 Model Size: {param_size:.2f} MB")
    logger.info(f"📊 Total Parameters: {all_param:,}")
    logger.info(f"📊 Trainable Parameters: {trainable_params:,}")
    logger.info(f"📊 Trainable Ratio: {100 * trainable_params / all_param:.2f}%")