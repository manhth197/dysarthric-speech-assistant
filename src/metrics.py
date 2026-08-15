"""
metrics.py - ENHANCED với Stratified Metrics
"""
import numpy as np
from typing import List, Dict, Union, Optional
import logging

try:
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logging.warning("⚠️  jiwer not installed. Run: pip install jiwer")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Helper: Ensure String List
# ═══════════════════════════════════════════════════════════════

def _ensure_string_list(texts: Union[List[str], List[List[str]], str]) -> List[str]:
    """Đảm bảo input luôn là List[str]"""
    if isinstance(texts, str):
        return [texts]
    
    if isinstance(texts, tuple):
        texts = list(texts)
    
    if not isinstance(texts, list):
        return [str(texts)]
    
    if not texts:
        return []
    
    result = []
    for item in texts:
        if isinstance(item, (list, tuple)):
            joined = " ".join(str(x) for x in item if x)
            result.append(joined)
        elif isinstance(item, str):
            result.append(item)
        else:
            result.append(str(item))
    
    return result


# ═══════════════════════════════════════════════════════════════
# CER & WER (Single Group)
# ═══════════════════════════════════════════════════════════════

def calculate_cer(
    predictions: Union[List[str], List[List[str]]], 
    references: Union[List[str], List[List[str]]]
) -> float:
    """Tính CER (0-100)"""
    if not JIWER_AVAILABLE:
        logger.error("❌ jiwer not available!")
        return 100.0
    
    try:
        preds = _ensure_string_list(predictions)
        refs = _ensure_string_list(references)
        
        valid_pairs = [
            (p.strip(), r.strip())
            for p, r in zip(preds, refs)
            if p and r and p.strip() and r.strip()
        ]
        
        if not valid_pairs:
            logger.warning("⚠️  No valid pairs for CER")
            return 100.0
        
        preds_clean, refs_clean = zip(*valid_pairs)
        preds_clean = list(preds_clean)
        refs_clean = list(refs_clean)
        
        cer_score = jiwer_cer(refs_clean, preds_clean) * 100
        
        if np.isnan(cer_score) or np.isinf(cer_score):
            logger.warning(f"⚠️  Invalid CER: {cer_score}")
            return 100.0
        
        return float(cer_score)
        
    except Exception as e:
        logger.error(f"❌ Error calculating CER: {e}")
        return 100.0


def calculate_wer(
    predictions: Union[List[str], List[List[str]]], 
    references: Union[List[str], List[List[str]]]
) -> float:
    """Tính WER (0-100)"""
    if not JIWER_AVAILABLE:
        logger.error("❌ jiwer not available!")
        return 100.0
    
    try:
        preds = _ensure_string_list(predictions)
        refs = _ensure_string_list(references)
        
        valid_pairs = [
            (p.strip(), r.strip())
            for p, r in zip(preds, refs)
            if p and r and p.strip() and r.strip()
        ]
        
        if not valid_pairs:
            logger.warning("⚠️  No valid pairs for WER")
            return 100.0
        
        preds_clean, refs_clean = zip(*valid_pairs)
        preds_clean = list(preds_clean)
        refs_clean = list(refs_clean)
        
        wer_score = jiwer_wer(refs_clean, preds_clean) * 100
        
        if np.isnan(wer_score) or np.isinf(wer_score):
            logger.warning(f"⚠️  Invalid WER: {wer_score}")
            return 100.0
        
        return float(wer_score)
        
    except Exception as e:
        logger.error(f"❌ Error calculating WER: {e}")
        return 100.0


# ═══════════════════════════════════════════════════════════════
# ✅ NEW: Stratified Metrics (Normal / Impaired / Global)
# ═══════════════════════════════════════════════════════════════

def calculate_stratified_metrics(
    predictions: List[str],
    references: List[str],
    is_impaired_flags: Optional[List[bool]] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    ✅ Tính metrics riêng cho Normal/Impaired/Global
    
    Args:
        predictions: Danh sách predictions
        references: Danh sách references
        is_impaired_flags: List[bool] - True nếu sample là impaired
        verbose: In log chi tiết
    
    Returns:
        {
            'global': {'wer': ..., 'cer': ..., 'n_samples': ...},
            'normal': {'wer': ..., 'cer': ..., 'n_samples': ...},
            'impaired': {'wer': ..., 'cer': ..., 'n_samples': ...},
            'gap': {'wer_gap': ..., 'cer_gap': ...}
        }
    """
    # Ensure lists
    preds = _ensure_string_list(predictions)
    refs = _ensure_string_list(references)
    
    # Nếu không có is_impaired_flags, coi tất cả là normal
    if is_impaired_flags is None:
        logger.warning("⚠️  is_impaired_flags is None, treating all as normal speech")
        is_impaired_flags = [False] * len(preds)
    
    # Ensure same length
    min_len = min(len(preds), len(refs), len(is_impaired_flags))
    preds = preds[:min_len]
    refs = refs[:min_len]
    is_impaired_flags = is_impaired_flags[:min_len]
    
    # ── Split by group ──
    normal_preds = []
    normal_refs = []
    impaired_preds = []
    impaired_refs = []
    
    for p, r, is_imp in zip(preds, refs, is_impaired_flags):
        if is_imp:
            impaired_preds.append(p)
            impaired_refs.append(r)
        else:
            normal_preds.append(p)
            normal_refs.append(r)
    
    # ── Calculate metrics ──
    results = {}
    
    # Global
    global_wer = calculate_wer(preds, refs)
    global_cer = calculate_cer(preds, refs)
    results['global'] = {
        'wer': round(float(global_wer), 2),
        'cer': round(float(global_cer), 2),
        'n_samples': len(preds)
    }
    
    # Normal
    if normal_preds:
        normal_wer = calculate_wer(normal_preds, normal_refs)
        normal_cer = calculate_cer(normal_preds, normal_refs)
        results['normal'] = {
            'wer': round(float(normal_wer), 2),
            'cer': round(float(normal_cer), 2),
            'n_samples': len(normal_preds)
        }
    else:
        results['normal'] = {'wer': 0.0, 'cer': 0.0, 'n_samples': 0}
    
    # Impaired
    if impaired_preds:
        impaired_wer = calculate_wer(impaired_preds, impaired_refs)
        impaired_cer = calculate_cer(impaired_preds, impaired_refs)
        results['impaired'] = {
            'wer': round(float(impaired_wer), 2),
            'cer': round(float(impaired_cer), 2),
            'n_samples': len(impaired_preds)
        }
    else:
        results['impaired'] = {'wer': 0.0, 'cer': 0.0, 'n_samples': 0}
    
    # Gap
    results['gap'] = {
        'wer_gap': round(results['impaired']['wer'] - results['normal']['wer'], 2),
        'cer_gap': round(results['impaired']['cer'] - results['normal']['cer'], 2),
    }
    
    # ── Logging ──
    if verbose:
        logger.info("=" * 60)
        logger.info("📊 STRATIFIED METRICS")
        logger.info("=" * 60)
        logger.info(f"🌐 GLOBAL   - WER: {results['global']['wer']:6.2f}%, CER: {results['global']['cer']:6.2f}% (n={results['global']['n_samples']})")
        logger.info(f"✅ NORMAL   - WER: {results['normal']['wer']:6.2f}%, CER: {results['normal']['cer']:6.2f}% (n={results['normal']['n_samples']})")
        logger.info(f"♿ IMPAIRED - WER: {results['impaired']['wer']:6.2f}%, CER: {results['impaired']['cer']:6.2f}% (n={results['impaired']['n_samples']})")
        logger.info(f"📏 GAP      - WER: {results['gap']['wer_gap']:+6.2f}%, CER: {results['gap']['cer_gap']:+6.2f}%")
        logger.info("=" * 60)
    
    return results


# ═══════════════════════════════════════════════════════════════════
# BLEU-4 Calculation (for GEC)
# ═══════════════════════════════════════════════════════════════════

def calculate_bleu(
    predictions: List[str], 
    references: List[str],
    tokenize: str = '13a',
    smooth_method: str = 'exp'
) -> float:
    """
    Tính BLEU-4 score với smooth7 method
    
    Args:
        predictions: List các chuỗi dự đoán
        references: List các chuỗi tham chiếu
        tokenize: Tokenization method ('13a' = Moses, 'char', 'intl', ...)
        smooth_method: Smoothing method ('exp' = smooth7, 'floor', 'add-k', ...)
    
    Returns:
        BLEU score (0-100)
    """
    if not SACREBLEU_AVAILABLE:
        # Fallback: Simple word-level BLEU-1
        return _calculate_simple_bleu(predictions, references)
    
    if not predictions or not references:
        return 0.0
    
    # ✅ FIX: Convert tuple to list
    if isinstance(predictions, tuple):
        predictions = list(predictions)
    if isinstance(references, tuple):
        references = list(references)
    
    # Loại bỏ các cặp rỗng
    valid_pairs = [
        (p.strip(), r.strip()) 
        for p, r in zip(predictions, references)
        if p and r and isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip()
    ]
    
    if not valid_pairs:
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    
    # ✅ FIX: Ensure preds and refs are lists
    preds = list(preds)
    refs = list(refs)
    
    try:
        bleu = SacreBLEU(
            tokenize=tokenize,
            smooth_method=smooth_method,
            smooth_value=None,
            effective_order=True
        )
        
        # ✅ sacrebleu expects: corpus_score(hypotheses: List[str], references: List[List[str]])
        result = bleu.corpus_score(preds, [refs])  # refs wrapped in list
        
        return float(result.score)  # Already 0-100
        
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        logger.error(f"Predictions type: {type(preds)}, References type: {type(refs)}")
        return 0.0


def _calculate_simple_bleu(predictions: List[str], references: List[str]) -> float:
    """Fallback: Simple word-level BLEU-1 (without sacrebleu)"""
    def bleu1_word(pred: str, ref: str) -> float:
        p_tokens = pred.split()
        r_set = set(ref.split())
        if not p_tokens:
            return 0.0
        return sum(1 for t in p_tokens if t in r_set) / len(p_tokens)
    
    valid_pairs = [
        (p.strip(), r.strip()) 
        for p, r in zip(predictions, references)
        if p.strip() and r.strip()
    ]
    
    if not valid_pairs:
        return 0.0
    
    scores = [bleu1_word(p, r) for p, r in valid_pairs]
    return float(np.mean(scores) * 100)  # Convert to 0-100


# ═══════════════════════════════════════════════════════════════════
# Token Accuracy & Exact Match (for GEC)
# ═══════════════════════════════════════════════════════════════════

def calculate_token_accuracy(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -100
) -> float:
    """
    Tính accuracy ở mức token
    
    Args:
        pred_ids: Predicted token IDs [batch_size, seq_len]
        label_ids: Label token IDs [batch_size, seq_len]
        ignore_index: Token ID to ignore (usually -100)
    
    Returns:
        Token accuracy (0-1)
    """
    # Align lengths
    pred_len = pred_ids.shape[1]
    label_len = label_ids.shape[1]
    common_len = min(pred_len, label_len)
    
    pred_aligned = pred_ids[:, :common_len]
    label_aligned = label_ids[:, :common_len]
    
    # Create valid mask
    valid_mask = label_aligned != ignore_index
    
    # Calculate accuracy
    correct = ((pred_aligned == label_aligned) & valid_mask).sum()
    total = valid_mask.sum()
    
    if total == 0:
        return 0.0
    
    return float(correct) / float(total)


def calculate_exact_match(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -100
) -> float:
    """
    Tính tỷ lệ sequence khớp hoàn toàn
    
    Args:
        pred_ids: Predicted token IDs [batch_size, seq_len]
        label_ids: Label token IDs [batch_size, seq_len]
        ignore_index: Token ID to ignore
    
    Returns:
        Exact match ratio (0-1)
    """
    # Align lengths
    pred_len = pred_ids.shape[1]
    label_len = label_ids.shape[1]
    common_len = min(pred_len, label_len)
    
    pred_aligned = pred_ids[:, :common_len]
    label_aligned = label_ids[:, :common_len]
    
    # Create valid mask
    valid_mask = label_aligned != ignore_index
    
    # Check if each sequence matches exactly
    per_seq_correct = (
        (pred_aligned == label_aligned) | ~valid_mask
    ).all(axis=-1)
    
    if len(per_seq_correct) == 0:
        return 0.0
    
    return float(per_seq_correct.mean())