"""
custom_trainer.py - FINAL FIX với Signature Compatibility
"""
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
import inspect

logger = logging.getLogger(__name__)


class WhisperTrainer(Seq2SeqTrainer):
    """
    ✅ FINAL FIX: Compatible với cả Transformers cũ và mới
    """
    
    # Custom keys (metadata only)
    CUSTOM_KEYS = ['is_impaired', 'transcripts', 'audio_ids']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_is_impaired = None
        self._last_transcripts = None
        self._last_audio_ids = None
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        ✅ ULTIMATE FIX: Lọc metadata + Ép kiểu Input về Half (FP16)
        """
        # 1. Lưu metadata cho compute_metrics (như sếp đã làm)
        for key in self.CUSTOM_KEYS:
            if key in inputs:
                setattr(self, f'_last_{key}', inputs[key])
        
        # 2. Lọc sạch metadata
        prepared = {k: v for k, v in inputs.items() if k not in self.CUSTOM_KEYS}
        
        # 3. Đẩy lên GPU (Parent handles device)
        prepared = super()._prepare_inputs(prepared)
        
        # ✅ 4. FIX DTYPE MISMATCH: Ép input về cùng kiểu với model (thường là Half)
        # Điều này giải quyết triệt để lỗi "Input float vs Bias Half" mà không tốn VRAM
        for k, v in prepared.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                prepared[k] = v.to(self.model.dtype)
                
        return prepared
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        ✅ FIXED: Compatible với cả old và new signature
        
        New Transformers (>= 4.35): training_step(model, inputs, num_items_in_batch)
        Old Transformers (< 4.35):  training_step(model, inputs)
        """
        # _prepare_inputs already called by parent, just call super
        if num_items_in_batch is not None:
            # New signature
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            # Old signature (fallback)
            return super().training_step(model, inputs)
    
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple:
        """
        ✅ prediction_step - _prepare_inputs already filters
        """
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        ✅ FIXED: Compatible với new signature
        
        New Transformers adds num_items_in_batch parameter
        """
        # _prepare_inputs already called by parent
        if num_items_in_batch is not None:
            # New signature
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        else:
            # Old signature
            return super().compute_loss(model, inputs, return_outputs)
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """Enhanced evaluation loop"""
        logger.info(f"🔍 Starting {description}...")
        
        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )
            
            logger.info(f"✅ {description} completed!")
            
            if hasattr(output, 'metrics') and output.metrics:
                for key, value in output.metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {key}: {value:.4f}")
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Error in evaluation_loop: {e}")
            logger.exception(e)
            raise