"""
train_asr.py - PRODUCTION VERSION với Stratified Metrics
"""
import os
os.environ["DISABLE_SAFETENSORS_AUTO_CONVERSION"] = "1"
import yaml
import torch
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
import logging

from .utils import (
    auto_detect_hardware,
    setup_logging,
    print_model_size,
)
from .model import PhoWhisperLoRA
from .dataset import CustomAudioDataset, collate_fn_asr
from .custom_trainer import WhisperTrainer
from .metrics import calculate_stratified_metrics

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Global Variables
# ═══════════════════════════════════════════════════════════════
processor = None
eval_dataset_global = None  # ✅ Store eval dataset for metrics


# ═══════════════════════════════════════════════════════════════
# Compute Metrics với Stratified Analysis
# ═══════════════════════════════════════════════════════════════

def compute_metrics_asr(eval_pred):
    """
    ✅ PRODUCTION: Compute stratified metrics (Normal/Impaired/Global)
    
    Fix: Lấy is_impaired từ eval_dataset thay vì từ batch
    """
    try:
        predictions, labels = eval_pred
        
        # 1. Unpack predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_ids = predictions
        label_ids = labels
        
        # 2. Handle -100 (ignored index)
        pred_ids = np.where(
            pred_ids != -100,
            pred_ids,
            processor.tokenizer.pad_token_id
        )
        label_ids = np.where(
            label_ids != -100,
            label_ids,
            processor.tokenizer.pad_token_id
        )
        
        # 3. Decode
        pred_str = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True
        )
        
        # 4. Flatten if nested
        def flatten(items):
            result = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    result.append(" ".join(str(x) for x in item if x))
                else:
                    result.append(str(item))
            return [s.strip() for s in result if s.strip()]
        
        pred_str = flatten(pred_str)
        label_str = flatten(label_str)
        
        # ✅ 5. Get is_impaired flags from eval_dataset (FIX!)
        is_impaired_flags = None
        
        if eval_dataset_global is not None:
            try:
                # Case 1: Subset (từ random_split)
                if hasattr(eval_dataset_global, 'indices'):
                    indices = eval_dataset_global.indices
                    base_dataset = eval_dataset_global.dataset
                    
                    # Extract is_impaired cho các indices trong eval set
                    is_impaired_flags = [
                        base_dataset.data[i]['is_impaired']
                        for i in indices
                    ]
                
                # Case 2: Direct dataset
                elif hasattr(eval_dataset_global, 'data'):
                    is_impaired_flags = [
                        item['is_impaired']
                        for item in eval_dataset_global.data
                    ]
                
                # ✅ Validation: Check length
                if is_impaired_flags and len(is_impaired_flags) != len(pred_str):
                    logger.warning(
                        f"⚠️  Length mismatch! "
                        f"is_impaired: {len(is_impaired_flags)}, "
                        f"predictions: {len(pred_str)}"
                    )
                    # Truncate to min length
                    min_len = min(len(is_impaired_flags), len(pred_str), len(label_str))
                    is_impaired_flags = is_impaired_flags[:min_len]
                    pred_str = pred_str[:min_len]
                    label_str = label_str[:min_len]
                
            except Exception as e:
                logger.error(f"❌ Error extracting is_impaired: {e}")
                logger.exception(e)
                is_impaired_flags = None
        
        if is_impaired_flags is None:
            logger.warning(
                "⚠️  Could not extract is_impaired flags. "
                "Treating all samples as normal speech."
            )
            is_impaired_flags = [False] * len(pred_str)
        
        # 6. Calculate stratified metrics
        metrics = calculate_stratified_metrics(
            predictions=pred_str,
            references=label_str,
            is_impaired_flags=is_impaired_flags,
            verbose=True  # Print detailed breakdown
        )
        
        # 7. Return flattened metrics for Trainer
        return {
            # Global (main metrics)
            'wer': metrics['global']['wer'],
            'cer': metrics['global']['cer'],
            
            # Normal
            'wer_normal': metrics['normal']['wer'],
            'cer_normal': metrics['normal']['cer'],
            'n_normal': metrics['normal']['n_samples'],
            
            # Impaired
            'wer_impaired': metrics['impaired']['wer'],
            'cer_impaired': metrics['impaired']['cer'],
            'n_impaired': metrics['impaired']['n_samples'],
            
            # Gap
            'wer_gap': metrics['gap']['wer_gap'],
            'cer_gap': metrics['gap']['cer_gap'],
        }
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in compute_metrics_asr: {e}")
        logger.exception(e)
        return {
            'wer': 100.0,
            'cer': 100.0,
            'wer_normal': 100.0,
            'cer_normal': 100.0,
            'wer_impaired': 100.0,
            'cer_impaired': 100.0,
            'wer_gap': 0.0,
            'cer_gap': 0.0,
            'n_normal': 0,
            'n_impaired': 0,
        }


# ═══════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════

def train_asr(config_path: str = "./config/config.yaml"):
    """
    ✅ PRODUCTION: Train ASR với stratified evaluation
    """
    global processor, eval_dataset_global
    
    # 1. Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. Hardware detection
    hardware_mode, hardware_info = auto_detect_hardware()
    hardware_config = config['hardware'][hardware_mode]
    
    # 3. Setup logging
    output_dir = config['asr']['output_dir']
    log_file = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("🚀 ASR TRAINING với STRATIFIED EVALUATION")
    logger.info("=" * 80)
    logger.info(f"🔧 Hardware: {hardware_mode}")
    logger.info(f"📝 Log file: {log_file}")
    
    # 4. Load model
    logger.info(f"📦 Loading model: {config['asr']['model_name']}")
    phowhisper = PhoWhisperLoRA(config['asr'], hardware_config)
    model = phowhisper.get_model()
    processor = phowhisper.get_processor()
    
    print_model_size(model)
    
    # 5. Load dataset
    logger.info("📂 Loading dataset...")
    full_dataset = CustomAudioDataset(
        metadata_path=config['dataset']['metadata_path'],
        audio_dir=config['dataset']['audio_dir'],
        processor=processor,
        sample_rate=config['dataset']['sample_rate'],
        max_audio_length=config['dataset']['max_audio_length'],
        normalize_text=config['vietnamese']['normalize_tone'],
    )
    
    # Split train/eval
    train_size = int(len(full_dataset) * config['dataset']['train_split'])
    eval_size = len(full_dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # ✅ Store eval_dataset globally for compute_metrics
    eval_dataset_global = eval_dataset
    
    logger.info(f"📊 Train samples: {len(train_dataset):,}")
    logger.info(f"📊 Eval samples:  {len(eval_dataset):,}")
    
    # ✅ Verify stratification in eval set
    if hasattr(eval_dataset, 'indices'):
        eval_impaired = sum(
            1 for i in eval_dataset.indices
            if eval_dataset.dataset.data[i]['is_impaired']
        )
        eval_normal = len(eval_dataset) - eval_impaired
        logger.info(f"📊 Eval distribution:")
        logger.info(f"   Normal:   {eval_normal} ({eval_normal/len(eval_dataset)*100:.1f}%)")
        logger.info(f"   Impaired: {eval_impaired} ({eval_impaired/len(eval_dataset)*100:.1f}%)")
    
    # 6. Training config
    train_cfg = config['asr']['training']
    
    # ✅ Validate hyperparameters
    lr = train_cfg.get('learning_rate', 1e-5)
    if lr > 1e-4:
        logger.warning(f"⚠️  Learning rate {lr:.2e} too high! Recommended: ≤ 1e-4")
    
    logger.info("=" * 80)
    logger.info("📌 TRAINING CONFIGURATION:")
    logger.info(f"  Learning Rate:    {lr:.2e}")
    logger.info(f"  Warmup Steps:     {train_cfg.get('warmup_steps', 500)}")
    logger.info(f"  Max Grad Norm:    {train_cfg.get('max_grad_norm', 1.0)}")
    logger.info(f"  Label Smoothing:  {train_cfg.get('label_smoothing_factor', 0.1)}")
    logger.info(f"  LR Scheduler:     {train_cfg.get('lr_scheduler_type', 'cosine')}")
    logger.info(f"  Num Epochs:       {train_cfg['num_epochs']}")
    logger.info(f"  Batch Size:       {hardware_config['batch_size']}")
    logger.info(f"  Grad Accumulation:{hardware_config['gradient_accumulation_steps']}")
    logger.info(f"  Effective Batch:  {hardware_config['batch_size'] * hardware_config['gradient_accumulation_steps']}")
    logger.info("=" * 80)
    
    # 7. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,
        
        # Batch
        per_device_train_batch_size=hardware_config['batch_size'],
        per_device_eval_batch_size=hardware_config['batch_size'],
        gradient_accumulation_steps=hardware_config['gradient_accumulation_steps'],
        
        # Learning
        num_train_epochs=train_cfg['num_epochs'],
        learning_rate=train_cfg['learning_rate'],
        warmup_steps=train_cfg.get('warmup_steps', 500),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        label_smoothing_factor=train_cfg.get('label_smoothing_factor', 0.1),
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine'),
        
        # FP16
        fp16=hardware_config.get('fp16', False) and not hardware_config.get('use_8bit', False),
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=train_cfg['eval_steps'],
        
        # Saving
        save_strategy="steps",
        save_steps=train_cfg['save_steps'],
        save_total_limit=train_cfg.get('save_total_limit', 3),
        load_best_model_at_end=True,
        
        # ✅ Metric for best model (global WER)
        metric_for_best_model="wer",
        greater_is_better=False,
        
        # Logging
        logging_steps=train_cfg.get('logging_steps', 50),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["tensorboard"],
        
        # DataLoader
        dataloader_num_workers=hardware_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=True,
        
        # Misc
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=train_cfg.get('push_to_hub', False),
    )
    
    # 8. Trainer
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_asr,
        compute_metrics=compute_metrics_asr,  # ✅ Stratified metrics
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.get('early_stopping_patience', 5)
            )
        ],
        processing_class=processor.feature_extractor,
    )
    
    # 9. Train
    logger.info("🚀 Starting training...")
    
    try:
        train_result = trainer.train()
        
        # 10. Save model
        logger.info(f"💾 Saving model to {output_dir}")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # 11. Final evaluation
        logger.info("📊 Final stratified evaluation...")
        eval_metrics = trainer.evaluate()
        
        # 12. Print results
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info("📊 FINAL STRATIFIED METRICS:")
        logger.info("-" * 80)
        logger.info(
            f"🌐 GLOBAL   - "
            f"WER: {eval_metrics.get('eval_wer', 0):6.2f}%, "
            f"CER: {eval_metrics.get('eval_cer', 0):6.2f}%"
        )
        logger.info(
            f"✅ NORMAL   - "
            f"WER: {eval_metrics.get('eval_wer_normal', 0):6.2f}%, "
            f"CER: {eval_metrics.get('eval_cer_normal', 0):6.2f}% "
            f"(n={eval_metrics.get('eval_n_normal', 0)})"
        )
        logger.info(
            f"♿ IMPAIRED - "
            f"WER: {eval_metrics.get('eval_wer_impaired', 0):6.2f}%, "
            f"CER: {eval_metrics.get('eval_cer_impaired', 0):6.2f}% "
            f"(n={eval_metrics.get('eval_n_impaired', 0)})"
        )
        logger.info("-" * 80)
        logger.info(
            f"📏 GAP      - "
            f"WER: {eval_metrics.get('eval_wer_gap', 0):+6.2f}%, "
            f"CER: {eval_metrics.get('eval_cer_gap', 0):+6.2f}%"
        )
        logger.info("=" * 80)
        
        # 13. Save metrics to JSON
        import json
        metrics_path = os.path.join(output_dir, "final_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"💾 Saved metrics to {metrics_path}")
        
        return trainer, eval_metrics
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    train_asr()