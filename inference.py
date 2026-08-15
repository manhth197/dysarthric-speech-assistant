import os
import re
import warnings
warnings.filterwarnings("ignore")
import transformers.safetensors_conversion as _sc
_sc.auto_conversion = lambda *a, **kw: None

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import yaml
import torch
import librosa
import argparse
import time
import transformers
transformers.logging.set_verbosity_error()
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from src.model import ViT5Corrector
from src.utils import normalize_vietnamese_text
from src.tts_model import VietnameseVITS

logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("__main__").setLevel(logging.WARNING)
logging.getLogger("src").setLevel(logging.WARNING)

def clean_gec_output(asr_text, gec_text):
    if not gec_text:
        return asr_text
        
    if re.search(r'[a-zA-Zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỵỷỹ]{8,}', gec_text):
        return asr_text
        
    words_asr = asr_text.split()
    words_gec = gec_text.split()
    
    if len(words_gec) < len(words_asr) * 0.7:
        return asr_text
        
    if "yễn" in gec_text and "nguyễn" in asr_text.lower():
        return asr_text
    gec_text = re.sub(r'\s+', ' ', gec_text).strip()
    return gec_text

class SpeechToTextPipeline:
    def __init__(self, config_path="./config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        asr_output_dir = self.config['asr']['output_dir']
        model_name = self.config['asr']['model_name']
        
        # Tham số sinh (đọc từ config; mặc định nhanh) — xem benchmark _bench_latency
        self.asr_num_beams = int(self.config['asr'].get('num_beams', 1))
        self.asr_max_length = int(self.config['asr'].get('max_length', 225))
        self.corr_num_beams = int(self.config['correction'].get('num_beams', 1))

        self.processor = WhisperProcessor.from_pretrained(model_name)

        base_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=False
        )
        
        self.asr_model = PeftModel.from_pretrained(base_model, asr_output_dir)
        self.asr_model = self.asr_model.merge_and_unload()
        self.asr_model.to(self.device)
        self.asr_model.eval()

        correction_output_dir = self.config['correction']['output_dir']
        self.corrector = ViT5Corrector(model_name=correction_output_dir)
        
        self.sample_rate = self.config['dataset']['sample_rate']
        
        tts_config = self.config.get('tts', {}) 
        self.tts = VietnameseVITS(config=tts_config)  

    def transcribe_and_correct(self, audio_path: str):
        if not os.path.exists(audio_path):
            return None, None

        start_time = time.time()

        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        input_features = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self.device)
        
        if self.device == "cuda":
            input_features = input_features.half()

        with torch.no_grad():
            predicted_ids = self.asr_model.generate(
                input_features,
                max_length=self.asr_max_length,
                num_beams=self.asr_num_beams,
                language="vi",
                task="transcribe"
            )
        
        raw_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        raw_text = normalize_vietnamese_text(raw_text)
        
        raw_gec_text = self.corrector.correct_text(raw_text, device=self.device, num_beams=self.corr_num_beams)
        corrected_text = clean_gec_output(raw_text, raw_gec_text)
        tts_text = re.sub(r'[^\w\s,.]', '', corrected_text)
        tts_text = " ".join(tts_text.split())
        print("\n" + "="*60)
        print("🎙️ KẾT QUẢ SUY DIỄN (INFERENCE RESULT)")
        print("="*60)
        print(f"🔹 Văn bản thô (ASR):   {raw_text}")
        print(f"🔸 Văn bản đã sửa (GEC): {corrected_text}")

        out_audio_path = self.tts.synthesize_to_file(corrected_text) 
        
        return raw_text, corrected_text, out_audio_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, default='./test.wav')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    args = parser.parse_args()

    pipeline = SpeechToTextPipeline(config_path=args.config)
    pipeline.transcribe_and_correct(args.audio)