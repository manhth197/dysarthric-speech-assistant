"""
src/tts_model.py
Tầng 3: Vietnamese TTS với VITS (Fixed version)
"""

import os
import re
import uuid
import unicodedata
import logging
from typing import List, Tuple
import wave

import numpy as np
import torch
import librosa  # pip install librosa

logger = logging.getLogger(__name__)


class VietnameseVITS:
    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "facebook/mms-tts-vie")
        self.output_audio_dir = config.get("output_audio_dir", "./outputs/tts_audio")
        self.max_chars = config.get("max_chars", 180)
        self.silence_ms_between_chunks = config.get("silence_ms_between_chunks", 150)
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = config.get("target_sample_rate", 44100)  # 🔧 Thêm config

        logger.info(f"🚀 Loading Vietnamese VITS: {self.model_name}")
        from transformers import AutoTokenizer, VitsModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = VitsModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = getattr(self.model.config, "sampling_rate", 16000)
        logger.info(f"✅ VITS loaded | native_sr={self.sample_rate} | target_sr={self.target_sample_rate} | device={self.device}")

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_text(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        if not text:
            return []

        if len(text) <= self.max_chars:
            return [text]

        parts = re.split(r'(?<=[\.\,\!\?\;\:])\s+', text)
        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(current) + len(part) + 1 <= self.max_chars:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)

                while len(part) > self.max_chars:
                    chunks.append(part[:self.max_chars].strip())
                    part = part[self.max_chars:].strip()

                current = part

        if current:
            chunks.append(current)

        return chunks

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Tổng hợp text -> waveform numpy (fixed version)
        """
        chunks = self._split_text(text)
        if not chunks:
            raise ValueError("Text rỗng, không thể tổng hợp.")

        audios = []
        
        # 🔧 Leading silence (200ms)
        leading_silence = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        audios.append(leading_silence)
        
        silence_between = np.zeros(
            int(self.sample_rate * self.silence_ms_between_chunks / 1000),
            dtype=np.float32
        )

        for i, chunk in enumerate(chunks):
            # 🔧 Skip quá ngắn
            if len(chunk.strip()) < 3:
                logger.warning(f"⚠️ Chunk {i+1} quá ngắn: '{chunk}', bỏ qua")
                continue
                
            inputs = self.tokenizer(chunk, return_tensors="pt").to(self.device)

            with torch.no_grad():
                try:
                    output = self.model(**inputs)
                    waveform = output.waveform.squeeze().cpu().float().numpy()
                    
                    if len(waveform) == 0:
                        logger.warning(f"⚠️ Waveform rỗng cho chunk {i+1}")
                        continue
                    
                    audios.append(waveform)
                    if i < len(chunks) - 1:  # Không thêm silence sau chunk cuối
                        audios.append(silence_between)
                        
                except Exception as e:
                    logger.error(f"❌ Lỗi chunk {i+1} '{chunk}': {e}")
                    continue

        # 🔧 Trailing silence (300ms)
        trailing_silence = np.zeros(int(self.sample_rate * 0.3), dtype=np.float32)
        audios.append(trailing_silence)

        if len(audios) <= 2:  # Chỉ có leading + trailing
            raise ValueError("Không có chunk nào được tổng hợp thành công")

        final_audio = np.concatenate(audios)
        
        # 🔧 Resample nếu cần
        if self.target_sample_rate != self.sample_rate:
            logger.info(f"🔄 Resampling {self.sample_rate}Hz -> {self.target_sample_rate}Hz")
            final_audio = librosa.resample(
                final_audio, 
                orig_sr=self.sample_rate, 
                target_sr=self.target_sample_rate
            )
        
        # 🔧 Normalize
        max_val = np.abs(final_audio).max()
        if max_val > 0:
            final_audio = final_audio / max_val * 0.95
        
        return final_audio, self.target_sample_rate

    def synthesize_to_file(self, text: str, output_path: str = None) -> str:
        """
        Tổng hợp và lưu ra file wav (browser-compatible)
        """
        if output_path is None:
            os.makedirs(self.output_audio_dir, exist_ok=True)
            output_path = os.path.join(self.output_audio_dir, f"tts_{uuid.uuid4().hex}.wav")
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        audio, sr = self.synthesize(text)
        
        # 🔧 Convert sang int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 🔧 Dùng wave module cho compatibility
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_int16.tobytes())

        logger.info(f"💾 Saved: {output_path} | {sr}Hz | {len(audio)/sr:.2f}s")
        return output_path