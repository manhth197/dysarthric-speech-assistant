"""
src/model.py
Model definitions: PhoWhisper với LoRA và ViT5 Corrector
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PhoWhisper + LoRA
# ──────────────────────────────────────────────────────────────────────────────

class PhoWhisperLoRA:
    """
    PhoWhisper ASR model với LoRA adaptation.

    config keys cần có:
        model_name          : str  – HuggingFace model ID
        lora.r              : int
        lora.lora_alpha     : int
        lora.target_modules : list[str]
        lora.lora_dropout   : float
        lora.bias           : str

    hardware_config keys:
        use_8bit                  : bool  (default False)
        fp16                      : bool  (default False)
        gradient_checkpointing    : bool  (default False)
    """

    def __init__(self, config: dict, hardware_config: dict) -> None:
        self.config          = config
        self.hardware_config = hardware_config
        self.model_name: str = config["model_name"]

        logger.info("🚀 Loading PhoWhisper: %s", self.model_name)

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            self.model_name
        )
        self.model = self._load_base_model()
        self.model = self._apply_lora(self.model)

        if hardware_config.get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_base_model(self) -> WhisperForConditionalGeneration:
        use_8bit: bool = self.hardware_config.get("use_8bit", False)

        if use_8bit:
            logger.info("🔧 Loading in 8-bit (low-VRAM mode)")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 8-bit loading. "
                    "Install with: pip install bitsandbytes"
                ) from exc

            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
                use_safetensors=False,
                torch_dtype=torch.float16,
            )
            model = prepare_model_for_kbit_training(model)

        else:
            dtype = (
                torch.float16
                if self.hardware_config.get("fp16", False)
                else torch.float32
            )
            logger.info("🔧 Loading in %s", dtype)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                use_safetensors=False,
            )

        return model

    def _apply_lora(
        self, model: WhisperForConditionalGeneration
    ) -> WhisperForConditionalGeneration:
        logger.info("🔧 Applying LoRA adapter …")

        lora_cfg = self.config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type="SEQ_2_SEQ_LM",
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def _enable_gradient_checkpointing(self) -> None:
        logger.info("🔧 Enabling gradient checkpointing")

        # Hỗ trợ cả PEFT wrapper lẫn model gốc
        base = getattr(self.model, "model", self.model)
        for part_name in ("encoder", "decoder"):
            part = getattr(base, part_name, None)
            if part is not None:
                part.gradient_checkpointing = True
                logger.debug("   gradient_checkpointing = True  [%s]", part_name)

    # ── public API ────────────────────────────────────────────────────────────

    def get_model(self):
        return self.model

    def get_processor(self) -> WhisperProcessor:
        return self.processor


# ──────────────────────────────────────────────────────────────────────────────
# ViT5 Corrector
# ──────────────────────────────────────────────────────────────────────────────

class ViT5Corrector:
    """
    ViT5-base model cho post-ASR text correction.

    Prefix chuẩn: ``"sửa lỗi: "``  — phải khớp CHÍNH XÁC với training data.
    Thay đổi TASK_PREFIX ở đây nếu bạn train lại với prefix khác.
    """


    TASK_PREFIX = "sửa lỗi: "

    # Tham số sinh mặc định — có thể override qua correct_text()
    _DEFAULT_MAX_LENGTH : int   = 256
    _DEFAULT_NUM_BEAMS  : int   = 4
    _DEFAULT_MIN_SIM    : float = 0.75

    def __init__(self, model_name: str = "vietai/vit5-base") -> None:
        self.model_name = model_name

        logger.info("🚀 Loading ViT5 Corrector: %s", model_name)

        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(model_name)
        )
        self.model.eval()  # default eval mode; train() khi fine-tune

        logger.info("✅ ViT5 Corrector loaded  |  prefix='%s'", self.TASK_PREFIX)

    # ── private helpers ───────────────────────────────────────────────────────

    def _to_device(self, device: str) -> None:
        """Chuyển model sang device nếu chưa ở đó."""
        current = next(self.model.parameters()).device
        if str(current) != device:
            self.model.to(device)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    # ── public API ────────────────────────────────────────────────────────────

    def get_model(self) -> T5ForConditionalGeneration:
        return self.model

    def get_tokenizer(self) -> T5Tokenizer:
        return self.tokenizer

    def correct_text(
        self,
        text: str,
        device: str = "cuda",
        min_similarity: float = _DEFAULT_MIN_SIM,
        max_length: int = _DEFAULT_MAX_LENGTH,
        num_beams: int = _DEFAULT_NUM_BEAMS,
    ) -> str:
        """
        Sửa lỗi chính tả / ASR noise cho một câu.

        Args:
            text           : ASR output cần sửa.
            device         : ``"cuda"`` hoặc ``"cpu"``.
            min_similarity : Ngưỡng similarity tối thiểu (0–1).
                             Nếu output của ViT5 khác input quá mức này
                             → giữ nguyên ASR output (tránh over-correction).
            max_length     : Độ dài token tối đa của output.
            num_beams      : Số beam cho beam search.

        Returns:
            Text đã sửa, hoặc ``text`` gốc nếu độ tin cậy thấp.
        """
        if not text or not text.strip():
            return text

        self._to_device(device)

        input_text = f"{self.TASK_PREFIX}{text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        corrected = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ── Confidence / sanity guard ─────────────────────────────────────────
        sim = self._similarity(text, corrected)

        if sim < min_similarity:
            logger.warning(
                "⚠️  ViT5 thay đổi quá lớn (sim=%.2f < %.2f) → giữ ASR output. "
                "input='%s'  output='%s'",
                sim, min_similarity, text, corrected,
            )
            return text

        logger.debug(
            "✅ Corrected (sim=%.2f) | '%s' → '%s'",
            sim, text, corrected,
        )
        return corrected

    def correct_batch(
        self,
        texts: list[str],
        device: str = "cuda",
        min_similarity: float = _DEFAULT_MIN_SIM,
        max_length: int = _DEFAULT_MAX_LENGTH,
        num_beams: int = _DEFAULT_NUM_BEAMS,
        batch_size: int = 16,
    ) -> list[str]:
        """
        Sửa lỗi cho một danh sách câu (batch inference).

        Args:
            texts      : Danh sách ASR output.
            batch_size : Số câu xử lý mỗi lần (tránh OOM).

        Returns:
            Danh sách câu đã sửa, cùng thứ tự với ``texts``.
        """
        if not texts:
            return []

        self._to_device(device)
        results: list[str] = []

        for start in range(0, len(texts), batch_size):
            batch_raw  = texts[start : start + batch_size]
            batch_in   = [f"{self.TASK_PREFIX}{t}" for t in batch_raw]

            inputs = self.tokenizer(
                batch_in,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            for raw, ids in zip(batch_raw, output_ids):
                corrected = self.tokenizer.decode(ids, skip_special_tokens=True)
                sim = self._similarity(raw, corrected)

                if sim < min_similarity:
                    logger.warning(
                        "⚠️  ViT5 over-correction (sim=%.2f) → giữ ASR. '%s'",
                        sim, raw,
                    )
                    results.append(raw)
                else:
                    results.append(corrected)

            logger.debug(
                "   Batch [%d:%d] done", start, start + len(batch_raw)
            )

        return results