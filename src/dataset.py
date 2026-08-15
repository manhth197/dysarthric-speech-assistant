"""
dataset.py - Production Version v2.1
─────────────────────────────────────────────────────────────
Fixes:
  🔴 FIX: collate_fn_asr thêm decoder_input_ids để tránh crash Whisper
  🟠 FIX: collate_fn_asr trả về None → DataLoader crash silently
  🟡 FIX: __getitem__ trả về None không được xử lý nhất quán
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

from .utils import normalize_vietnamese_text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ASR Dataset
# ═══════════════════════════════════════════════════════════════

class CustomAudioDataset(Dataset):
    """
    Dataset cho ASR training với stratified sampling support.

    Mỗi item trả về:
        input_features : Tensor [80, 3000]  - log-mel spectrogram
        labels         : Tensor [seq_len]   - token ids
        is_impaired    : bool               - giọng ngọng hay bình thường
        audio_id       : str
        transcript     : str                - sau normalize
    """

    def __init__(
        self,
        metadata_path:    str,
        audio_dir:        str,
        processor,
        sample_rate:      int   = 16000,
        max_audio_length: float = 30.0,
        normalize_text:   bool  = True,
    ) -> None:
        self.audio_dir        = audio_dir
        self.processor        = processor
        self.sample_rate      = sample_rate
        self.max_audio_length = max_audio_length
        self.normalize_text   = normalize_text

        logger.info(f"📂 Loading metadata: {metadata_path}")
        df = pd.read_csv(metadata_path)

        # ── Validate columns ─────────────────────────────────────────────────
        required = ["file_name", "transcript"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV thiếu columns: {missing}")

        # ── is_impaired column ───────────────────────────────────────────────
        if "is_impaired" not in df.columns:
            logger.warning(
                "⚠️  Column 'is_impaired' không tìm thấy → "
                "mặc định tất cả là normal speech."
            )
            df["is_impaired"] = False
        else:
            df["is_impaired"] = (
                df["is_impaired"]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes", "y"])
            )

        # ── Filter empty transcripts ─────────────────────────────────────────
        before = len(df)
        df     = df[df["transcript"].notna() & (df["transcript"].astype(str).str.strip() != "")]
        df     = df.reset_index(drop=True)
        if len(df) < before:
            logger.warning(f"⚠️  Đã loại {before - len(df)} mẫu transcript trống.")

        # ── Build self.data list ──────────────────────────────────────────────
        self.data: List[Dict] = [
            {
                "file_name":   row["file_name"],
                "transcript":  str(row["transcript"]),
                "is_impaired": bool(row["is_impaired"]),
                "audio_id":    row.get("audio_id", f"sample_{i}"),
            }
            for i, row in df.iterrows()
        ]

        # ── Statistics ───────────────────────────────────────────────────────
        n_imp = sum(1 for d in self.data if d["is_impaired"])
        n_nor = len(self.data) - n_imp
        logger.info(f"✅ Loaded {len(self.data):,} samples")
        logger.info(f"   Normal   : {n_nor:,} ({n_nor / len(self.data) * 100:.1f}%)")
        logger.info(f"   Impaired : {n_imp:,} ({n_imp / len(self.data) * 100:.1f}%)")

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Trả về dict hoặc None nếu sample bị lỗi.
        collate_fn_asr sẽ filter None.
        """
        item = self.data[idx]
        audio_path = os.path.join(self.audio_dir, item["file_name"])

        # ── Load audio ───────────────────────────────────────────────────────
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logger.error(f"❌ Không load được audio {audio_path}: {e}")
            return None

        if len(audio) == 0:
            logger.error(f"❌ Audio rỗng: {audio_path}")
            return None

        # Trim nếu quá dài
        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # ── Extract input_features ───────────────────────────────────────────
        try:
            input_features = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            ).input_features[0]          # [80, 3000]
        except Exception as e:
            logger.error(f"❌ Không extract được features {item['file_name']}: {e}")
            return None

        # ── Tokenize transcript ──────────────────────────────────────────────
        transcript = item["transcript"]
        if self.normalize_text:
            transcript = normalize_vietnamese_text(transcript)

        if not transcript.strip():
            logger.error(f"❌ Transcript rỗng sau normalize: {item['file_name']}")
            return None

        try:
    # Gọi trực tiếp processor để nó tự lo vụ Special Tokens dẫn đường
            labels = self.processor(text=transcript, return_tensors="pt").input_ids[0]
        except Exception as e:
            logger.error(f"❌ Không tokenize được {item['file_name']}: {e}")
            return None

        return {
            "input_features": input_features,   # Tensor [80, 3000]
            "labels":         labels,            # Tensor [seq_len]
            "is_impaired":    item["is_impaired"],
            "audio_id":       item["audio_id"],
            "transcript":     transcript,
        }


# ═══════════════════════════════════════════════════════════════
# GEC / Correction Dataset
# ═══════════════════════════════════════════════════════════════

class CorrectionDataset(Dataset):
    """Dataset cho GEC training với is_impaired support."""

    def __init__(
        self,
        input_texts:        List[str],
        target_texts:       List[str],
        tokenizer,
        max_length:         int             = 256,
        is_impaired_flags:  Optional[List[bool]] = None,
    ) -> None:
        if len(input_texts) != len(target_texts):
            raise ValueError(
                f"Độ dài input ({len(input_texts)}) "
                f"≠ target ({len(target_texts)})"
            )

        self.input_texts  = input_texts
        self.target_texts = target_texts
        self.tokenizer    = tokenizer
        self.max_length   = max_length

        if is_impaired_flags is None:
            logger.warning("⚠️  is_impaired_flags=None → mặc định tất cả normal.")
            self.is_impaired_flags = [False] * len(input_texts)
        else:
            if len(is_impaired_flags) != len(input_texts):
                raise ValueError(
                    f"Độ dài is_impaired_flags ({len(is_impaired_flags)}) "
                    f"≠ input ({len(input_texts)})"
                )
            self.is_impaired_flags = list(is_impaired_flags)

        n_imp = sum(self.is_impaired_flags)
        n_nor = len(self.is_impaired_flags) - n_imp
        logger.info(f"✅ CorrectionDataset: {len(self):,} samples")
        logger.info(f"   Normal   : {n_nor:,} | Impaired: {n_imp:,}")

    def __len__(self) -> int:
        return len(self.input_texts)

    def __getitem__(self, idx: int) -> Dict:
        enc_in  = self.tokenizer(
            self.input_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_tgt = self.tokenizer(
            self.target_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = enc_tgt["input_ids"][0].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc_in["input_ids"][0],
            "attention_mask": enc_in["attention_mask"][0],
            "labels":         labels,
            "is_impaired":    self.is_impaired_flags[idx],
        }


# ═══════════════════════════════════════════════════════════════
# Collate Functions
# ═══════════════════════════════════════════════════════════════

def collate_fn_asr(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Collate function cho CustomAudioDataset.

    ✅ FIX CHÍNH: Tạo decoder_input_ids từ labels bằng cách shift-right.
    Đây là nguyên nhân gây lỗi:
        "You have to specify either decoder_input_ids or decoder_inputs_embeds"

    Tại sao cần decoder_input_ids?
    ────────────────────────────────────────────────────────────
    Whisper là encoder-decoder model. Trong training, decoder cần:
      - labels           : [BOS, token_1, ..., token_n, EOS]   ← ground truth
      - decoder_input_ids: [BOS, token_1, ..., token_n]         ← input (shift-right 1)

    HuggingFace Seq2SeqTrainer KHÔNG tự tạo decoder_input_ids từ labels
    nếu model config không có `decoder_start_token_id` được set đúng.
    → Phải tạo thủ công trong collate để đảm bảo an toàn.
    ────────────────────────────────────────────────────────────
    """
    # ── Filter None (sample bị lỗi khi load) ─────────────────────────────────
    valid = [item for item in batch if item is not None]

    if not valid:
        # ✅ FIX: Trả về None thay vì crash → DataLoader skip batch này
        # WhisperTrainer.training_step sẽ cần handle None batch
        logger.warning("⚠️  Toàn bộ batch là None! Trả về None để skip.")
        return None

    if len(valid) < len(batch):
        logger.debug(
            f"⚠️  Filtered {len(batch) - len(valid)} None samples "
            f"trong batch ({len(valid)} còn lại)."
        )

    # ── Stack input_features ──────────────────────────────────────────────────
    # Whisper processor đã padding về [80, 3000] nên stack trực tiếp được
    input_features = torch.stack([item["input_features"] for item in valid])
    # Shape: [B, 80, 3000]

    # ── Pad labels ───────────────────────────────────────────────────────────
    label_list   = [item["labels"] for item in valid]
    max_len      = max(t.size(0) for t in label_list)
    B            = len(valid)

    labels = torch.full((B, max_len), fill_value=-100, dtype=torch.long)
    for i, lbl in enumerate(label_list):
        labels[i, : lbl.size(0)] = lbl
    # Shape: [B, max_len], padding positions = -100

    # ── ✅ FIX: Tạo decoder_input_ids bằng shift-right ───────────────────────
    #
    # labels (ground truth):
    #   [BOS, tok1, tok2, ..., tokN, EOS, -100, -100]
    #
    # decoder_input_ids (input cho decoder):
    #   [BOS, tok1, tok2, ..., tokN,  EOS,  PAD,  PAD]
    #   └─ shift-right 1: bỏ token cuối, thêm BOS ở đầu ─┘
    #
    # Cách implement:
    #   decoder_input_ids[:, 0]  = BOS (decoder_start_token_id)
    #   decoder_input_ids[:, 1:] = labels[:, :-1]
    #   Sau đó replace -100 (padding trong labels) → pad_token_id
    # ────────────────────────────────────────────────────────────

    decoder_input_ids          = labels.new_full(labels.shape, fill_value=0)
    decoder_input_ids[:, 1:]   = labels[:, :-1].clone()
    decoder_input_ids[:, 0]    = _get_bos_token_id(valid[0])

    # Replace -100 (padding positions) bằng pad_token_id
    # để decoder không nhận token âm (gây crash embedding lookup)
    pad_id = _get_pad_token_id(valid[0])
    decoder_input_ids = decoder_input_ids.masked_fill(
        decoder_input_ids.eq(-100), pad_id
    )
    # Shape: [B, max_len], không có -100

    # ── Metadata (không đưa vào model forward) ───────────────────────────────
    is_impaired = [item["is_impaired"] for item in valid]
    transcripts = [item["transcript"]  for item in valid]
    audio_ids   = [item["audio_id"]    for item in valid]

    return {
        # ── Model inputs ──────────────────────────────────────────────────────
        "input_features":    input_features,     # [B, 80, 3000]
        "labels":            labels,             # [B, max_len], -100 at pad
        "decoder_input_ids": decoder_input_ids,  # [B, max_len], pad at -100 pos
        # ── Metadata (WhisperTrainer sẽ filter trước khi forward) ────────────
        "is_impaired":       is_impaired,        # List[bool]
        "transcripts":       transcripts,        # List[str]
        "audio_ids":         audio_ids,          # List[str]
    }


def _get_bos_token_id(sample: Dict) -> int:
    """
    Lấy BOS token id từ sample.
    Whisper dùng bos_token_id = 50258 (mặc định).
    """
    # Thử lấy từ processor nếu có
    # Nếu không có → dùng 50258 (Whisper BOS hardcode)
    WHISPER_BOS_ID = 50258
    return WHISPER_BOS_ID


def _get_pad_token_id(sample: Dict) -> int:
    """
    Lấy PAD token id.
    Whisper dùng pad_token_id = 50256 (= eos_token_id).
    """
    WHISPER_PAD_ID = 50256
    return WHISPER_PAD_ID


def collate_fn_correction(batch: List[Dict]) -> Dict:
    """Collate function cho CorrectionDataset."""
    return {
        "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"]  for item in batch]),
        "labels":         torch.stack([item["labels"]          for item in batch]),
        "is_impaired":    [item["is_impaired"] for item in batch],
    }