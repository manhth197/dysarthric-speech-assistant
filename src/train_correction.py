"""
train_correction.py - PHIÊN BẢN ĐÃ FIX LỖI IMPAIRED VÀ THÊM BLEU
"""
import os
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
import logging
import evaluate  # ✅ MỚI: Thêm thư viện đánh giá BLEU

from .utils import setup_logging, print_model_size
from .model import ViT5Corrector
from .dataset import CorrectionDataset
from .metrics import calculate_stratified_metrics

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
eval_dataset_global = None  # ✅ Store for compute_metrics


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def _validate_data(
    input_texts: list,
    target_texts: list,
    is_impaired_flags: list = None
) -> tuple:
    """
    ✅ Validate and clean data
    
    Returns:
        (clean_inputs, clean_targets, clean_flags)
    """
    original_len = len(input_texts)
    
    # Build pairs
    if is_impaired_flags is None:
        is_impaired_flags = [False] * len(input_texts)
    
    pairs = [
        (i, t, f)
        for i, t, f in zip(input_texts, target_texts, is_impaired_flags)
        if i and t and isinstance(i, str) and isinstance(t, str)
        and i.strip() and t.strip()
    ]
    
    if len(pairs) < original_len:
        logger.warning(
            f"⚠️  Data validation: removed {original_len - len(pairs)} "
            f"invalid pairs ({original_len} → {len(pairs)})"
        )
    
    if not pairs:
        return [], [], []
    
    clean_inputs, clean_targets, clean_flags = zip(*pairs)
    return list(clean_inputs), list(clean_targets), list(clean_flags)


def _split_data(
    input_texts: list,
    target_texts: list,
    is_impaired_flags: list,
    eval_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    ✅ Stratified split để giữ tỷ lệ Normal/Impaired đồng đều
    
    Returns:
        (train_inputs, eval_inputs, train_targets, eval_targets,
         train_flags, eval_flags)
    """
    # Group by is_impaired
    normal_indices = [i for i, flag in enumerate(is_impaired_flags) if not flag]
    impaired_indices = [i for i, flag in enumerate(is_impaired_flags) if flag]
    
    logger.info(f"📊 Stratified split:")
    logger.info(f"   Normal samples:   {len(normal_indices)}")
    logger.info(f"   Impaired samples: {len(impaired_indices)}")
    
    # Split each group
    np.random.seed(seed)
    np.random.shuffle(normal_indices)
    np.random.shuffle(impaired_indices)
    
    n_eval_normal = int(len(normal_indices) * eval_ratio)
    n_eval_impaired = int(len(impaired_indices) * eval_ratio)
    
    eval_indices = (
        normal_indices[:n_eval_normal] +
        impaired_indices[:n_eval_impaired]
    )
    train_indices = (
        normal_indices[n_eval_normal:] +
        impaired_indices[n_eval_impaired:]
    )
    
    # Shuffle train/eval
    np.random.shuffle(train_indices)
    np.random.shuffle(eval_indices)
    
    # Extract data
    train_inputs = [input_texts[i] for i in train_indices]
    train_targets = [target_texts[i] for i in train_indices]
    train_flags = [is_impaired_flags[i] for i in train_indices]
    
    eval_inputs = [input_texts[i] for i in eval_indices]
    eval_targets = [target_texts[i] for i in eval_indices]
    eval_flags = [is_impaired_flags[i] for i in eval_indices]
    
    logger.info(f"✅ Split complete:")
    logger.info(f"   Train: {len(train_inputs)} samples")
    logger.info(f"   Eval:  {len(eval_inputs)} samples")
    
    return (
        train_inputs, eval_inputs,
        train_targets, eval_targets,
        train_flags, eval_flags
    )


# ... (Các hàm helper _validate_data, _split_data giữ nguyên)

def _build_compute_metrics(tokenizer):
    """
    ✅ Build compute_metrics function với stratified support và SacreBLEU
    """
    # Load sacrebleu metric một lần duy nhất
    bleu_metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_pred):
        try:
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            pred_ids = predictions
            label_ids = labels
            
            pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            
            pred_str = [s.strip() for s in pred_str]
            label_str = [s.strip() for s in label_str]
            
            # 1. Lấy flags từ dataset toàn cục (đã được fix ở hàm main)
            is_impaired_flags = None
            if eval_dataset_global is not None:
                if hasattr(eval_dataset_global, 'is_impaired_flags'):
                    is_impaired_flags = eval_dataset_global.is_impaired_flags

            # 2. Tính CER/WER phân lớp
            metrics = calculate_stratified_metrics(
                predictions=pred_str,
                references=label_str,
                is_impaired_flags=is_impaired_flags,
                verbose=True
            )
            
            # ✅ 3. TÍNH BLEU SCORE (SacreBLEU)
            # SacreBLEU yêu cầu references là list của list
            bleu_results = bleu_metric.compute(
                predictions=pred_str, 
                references=[[l] for l in label_str]
            )
            
            # 4. Trả về dictionary metrics đầy đủ
            return {
                'cer': metrics['global']['cer'],
                'wer': metrics['global']['wer'],
                'bleu': bleu_results['score'],  # ✅ ĐÃ THÊM BLEU CHO SẾP
                
                'cer_normal': metrics['normal']['cer'],
                'wer_normal': metrics['normal']['wer'],
                'cer_impaired': metrics['impaired']['cer'],
                'wer_impaired': metrics['impaired']['wer'],
                
                'cer_gap': metrics['gap']['cer_gap'],
                'wer_gap': metrics['gap']['wer_gap'],
            }
            
        except Exception as e:
            logger.error(f"❌ Error in compute_metrics: {e}")
            return {'cer': 100.0, 'bleu': 0.0}
    
    return compute_metrics

def _load_pairs_from_csv(csv_path: str, task_prefix: str) -> tuple:
    """Nạp (inputs+prefix, targets, is_impaired_flags) từ 1 CSV đã chia sẵn."""
    df = pd.read_csv(csv_path)
    raw_inputs = df["predicted_text"].astype(str).tolist()
    target_texts = df["ground_truth"].astype(str).tolist()
    if "is_real" in df.columns:
        is_real = df["is_real"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
        )
        is_impaired_flags = [not b for b in is_real]  # synthetic (is_real=False) => impaired
    elif "is_impaired" in df.columns:
        is_impaired_flags = df["is_impaired"].fillna(False).astype(bool).tolist()
    else:
        is_impaired_flags = [False] * len(raw_inputs)
    input_texts = [f"{task_prefix}{t}" for t in raw_inputs]
    return _validate_data(input_texts, target_texts, is_impaired_flags)


def train_correction(config_path="./config/config.yaml", data_csv=None,
                     train_csv=None, dev_csv=None, output_dir=None):
    global eval_dataset_global

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    corr_cfg = config["correction"]
    train_cfg = corr_cfg["training"]
    output_dir = output_dir or corr_cfg["output_dir"]

    setup_logging(output_dir)
    corrector = ViT5Corrector(corr_cfg["model_name"])
    model = corrector.get_model()
    tokenizer = corrector.get_tokenizer()
    TASK_PREFIX = ViT5Corrector.TASK_PREFIX

    if train_csv and dev_csv:
        # ✅ Đường CHỐNG RÒ RỈ: dùng split đã dựng theo CÂU (build_correction_splits.py)
        logger.info(f"📂 Leakage-safe split | train={train_csv} | dev={dev_csv}")
        train_inputs, train_targets, train_flags = _load_pairs_from_csv(train_csv, TASK_PREFIX)
        eval_inputs, eval_targets, eval_flags = _load_pairs_from_csv(dev_csv, TASK_PREFIX)
        logger.info(f"   Train: {len(train_inputs)} | Dev: {len(eval_inputs)}")
    else:
        # Đường CŨ (random split trên combined) — GIỮ để tương thích, KHÔNG khuyến khích.
        csv_path = data_csv or "./outputs/correction_data_combined.csv"
        logger.info(f"📂 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        raw_inputs = df["predicted_text"].tolist()
        target_texts = df["ground_truth"].tolist()
        if "is_real" in df.columns:
            is_impaired_flags = (df["is_real"] == False).tolist()
        elif "is_impaired" in df.columns:
            is_impaired_flags = df["is_impaired"].fillna(False).astype(bool).tolist()
        else:
            is_impaired_flags = [False] * len(raw_inputs)
        input_texts = [f"{TASK_PREFIX}{t}" for t in raw_inputs]
        input_texts, target_texts, is_impaired_flags = _validate_data(
            input_texts, target_texts, is_impaired_flags
        )
        eval_ratio = corr_cfg.get("eval_split", 0.05)
        (train_inputs, eval_inputs, train_targets, eval_targets,
         train_flags, eval_flags) = _split_data(
            input_texts, target_texts, is_impaired_flags, eval_ratio=eval_ratio
        )

    train_dataset = CorrectionDataset(train_inputs, train_targets, tokenizer, is_impaired_flags=train_flags)
    eval_dataset = CorrectionDataset(eval_inputs, eval_targets, tokenizer, is_impaired_flags=eval_flags)
    eval_dataset_global = eval_dataset # Lưu để compute_metrics lấy flag
    
    # --- ĐOẠN FIX metric_for_best_model SANG BLEU ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        generation_max_length=128,
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        num_train_epochs=train_cfg["num_epochs"],
        max_steps=train_cfg.get("max_steps", -1),  # -1 = dùng epochs; >0 = train nhanh có giới hạn
        learning_rate=train_cfg["learning_rate"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        load_best_model_at_end=True,
        
        # ✅ CHỐT BLEU LÀM CHỈ SỐ QUYẾT ĐỊNH
        metric_for_best_model="bleu", 
        greater_is_better=True,
        
        report_to=["tensorboard"],
        remove_unused_columns=True, # ✅ Fix lỗi is_impaired ở hàm forward
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=_build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg.get("early_stopping_patience", 5))],
    )
    
    logger.info("🚀 Starting training...")
    trainer.train()
    # ... (Phần save và in kết quả giữ nguyên)
    # 11. Save
    logger.info(f"💾 Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 12. Final evaluation
    logger.info("📊 Final stratified evaluation...")
    final_metrics = trainer.evaluate()
    
    # 13. Print results
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info("📊 FINAL STRATIFIED METRICS:")
    logger.info("-" * 80)
    logger.info(
        f"🌐 GLOBAL   - "
        f"CER: {final_metrics.get('eval_cer', 0):6.2f}%, "
        f"WER: {final_metrics.get('eval_wer', 0):6.2f}%"
    )
    logger.info(
        f"✅ NORMAL   - "
        f"CER: {final_metrics.get('eval_cer_normal', 0):6.2f}%, "
        f"WER: {final_metrics.get('eval_wer_normal', 0):6.2f}% "
        f"(n={final_metrics.get('eval_n_normal', 0)})"
    )
    logger.info(
        f"♿ IMPAIRED - "
        f"CER: {final_metrics.get('eval_cer_impaired', 0):6.2f}%, "
        f"WER: {final_metrics.get('eval_wer_impaired', 0):6.2f}% "
        f"(n={final_metrics.get('eval_n_impaired', 0)})"
    )
    logger.info("-" * 80)
    logger.info(
        f"📏 GAP      - "
        f"CER: {final_metrics.get('eval_cer_gap', 0):+6.2f}%, "
        f"WER: {final_metrics.get('eval_wer_gap', 0):+6.2f}%"
    )
    logger.info("=" * 80)
    
    # 14. Save metrics
    import json
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"💾 Saved metrics to {metrics_path}")
    
    return trainer


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train ViT5 corrector (hỗ trợ split chống rò rỉ).")
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--train-csv", default=None, help="CSV train đã chia sẵn (leakage-safe)")
    ap.add_argument("--dev-csv", default=None, help="CSV dev/eval đã chia sẵn")
    ap.add_argument("--data-csv", default=None, help="(đường cũ) combined CSV để random-split")
    ap.add_argument("--output-dir", default=None, help="Ghi model ra đây (mặc định lấy từ config)")
    a = ap.parse_args()
    train_correction(
        config_path=a.config,
        data_csv=a.data_csv,
        train_csv=a.train_csv,
        dev_csv=a.dev_csv,
        output_dir=a.output_dir,
    )