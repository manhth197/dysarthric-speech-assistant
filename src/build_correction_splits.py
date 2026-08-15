"""
src/build_correction_splits.py — Dựng tập train/eval CHỐNG RÒ RỈ cho ViT5 Corrector.

VÌ SAO CẦN (đã verify trên code + CSV thật):
  - `correction_data_combined.csv` = 366 real (= 122 cặp lỗi × real_weight 3) + 758.739 synthetic.
  - `train_correction._split_data` chia NGẪU NHIÊN theo dòng ⇒ 3 bản sao của mỗi cặp real,
    và các biến thể synthetic CÙNG một câu gốc, rơi vào cả train lẫn eval ⇒ BLEU lạc quan.
  - Eval gần như 100% synthetic ⇒ BLEU 87,58 chủ yếu đo "tự đảo lại bộ hàm noise của chính nó",
    KHÔNG đo sửa lỗi dysarthric thật.
  - Corrector không hề thấy ví dụ "input đã đúng → giữ nguyên" ⇒ rủi ro over-correction
    (chỉ được chặn ở inference bằng Sanity Guard).

FILE NÀY dựng split theo NHÓM = CÂU (ground_truth), bảo đảm:
  1. Không câu nào (kể cả biến thể synthetic của nó) nằm cả train lẫn eval.
  2. Ba tập eval TÁCH BẠCH để đo 3 câu hỏi khác nhau:
       - eval_real_error : cặp ASR-nghe-sai THẬT trên câu chưa train  → đo SỬA LỖI THẬT.
       - eval_keep_correct: câu ASR nghe ĐÚNG (input == target)        → đo OVER-CORRECTION.
       - eval_synthetic  : cặp synthetic trên câu chưa train           → dán nhãn rõ "synthetic".
  3. Train THÊM cặp "giữ đúng" (identity: input == target) để corrector học "đúng thì đừng sửa".

KẾT QUẢ (mặc định) ghi vào outputs/correction_splits/:
  train.csv, dev.csv, eval_real_error.csv, eval_keep_correct.csv, eval_synthetic.csv, report.json

CÁCH DÙNG:
  python -m src.build_correction_splits
  python -m src.build_correction_splits --err-eval-ratio 0.3 --keep-correct-eval 300 --syn-eval 500
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

STD_COLS = ["predicted_text", "ground_truth", "is_real", "severity", "cp_type", "kind"]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))


def _std(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Chuẩn hoá về STD_COLS (điền cột thiếu)."""
    out = pd.DataFrame()
    out["predicted_text"] = df["predicted_text"].astype(str)
    out["ground_truth"] = df["ground_truth"].astype(str)
    out["is_real"] = _as_bool(df["is_real"]) if "is_real" in df.columns else False
    out["severity"] = df["severity"] if "severity" in df.columns else "unknown"
    out["cp_type"] = df["cp_type"] if "cp_type" in df.columns else "unknown"
    out["kind"] = kind
    return out[STD_COLS]


def build_splits(
    real_csv: str = "./outputs/correction_data_real.csv",
    synthetic_csv: str = "./outputs/correction_data_synthetic.csv",
    out_dir: str = "./outputs/correction_splits",
    err_eval_ratio: float = 0.30,      # tỷ lệ CÂU-lỗi-thật giữ cho eval_real_error
    keep_correct_eval: int = 300,      # số CÂU đúng giữ cho eval_keep_correct
    syn_eval_sentences: int = 500,     # số CÂU synthetic-thuần giữ cho eval_synthetic
    syn_eval_per_sentence: int = 2,    # số biến thể/câu trong eval_synthetic
    real_weight: int = 3,              # nhân bản cặp real (giữ đúng thói quen cũ)
    keep_correct_train: int = 6000,    # số cặp "giữ đúng" từ câu THẬT thêm vào train (trước ×weight)
    clean_identity_train: int = 20000, # số cặp clean→clean từ câu synthetic thêm vào train
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    real = pd.read_csv(real_csv, encoding="utf-8-sig")
    real["is_real"] = _as_bool(real["is_real"]) if "is_real" in real.columns else True
    real["predicted_text"] = real["predicted_text"].astype(str)
    real["ground_truth"] = real["ground_truth"].astype(str)

    real_err = real[real["predicted_text"] != real["ground_truth"]].copy()   # ASR nghe SAI
    real_ok = real[real["predicted_text"] == real["ground_truth"]].copy()    # ASR nghe ĐÚNG

    err_sents = sorted(real_err["ground_truth"].unique().tolist())
    ok_sents = sorted(real_ok["ground_truth"].unique().tolist())

    # ── 1) Chọn CÂU cho eval_real_error (giữ theo câu, không theo dòng) ──
    err_perm = err_sents.copy()
    rng.shuffle(err_perm)
    n_err_eval = max(1, int(round(len(err_perm) * err_eval_ratio)))
    eval_err_sents = set(err_perm[:n_err_eval])

    # ── 2) Chọn CÂU cho eval_keep_correct (câu ASR đúng, KHÔNG trùng câu lỗi) ──
    ok_only = [s for s in ok_sents if s not in set(err_sents)]
    rng.shuffle(ok_only)
    eval_keep_sents = set(ok_only[:keep_correct_eval])

    # ── 3) Đọc synthetic (chunk để nhẹ RAM), tách theo câu ──
    syn = pd.read_csv(synthetic_csv, encoding="utf-8-sig")
    syn["is_real"] = _as_bool(syn["is_real"]) if "is_real" in syn.columns else False
    syn["predicted_text"] = syn["predicted_text"].astype(str)
    syn["ground_truth"] = syn["ground_truth"].astype(str)

    syn_sents_all = syn["ground_truth"].unique().tolist()
    # câu synthetic-thuần: không phải câu lỗi-thật, không phải câu giữ-đúng eval
    reserved = set(err_sents) | eval_keep_sents
    syn_pure = [s for s in syn_sents_all if s not in reserved]
    rng.shuffle(syn_pure)
    eval_syn_sents = set(syn_pure[:syn_eval_sentences])

    # Tập câu CẤM train (mọi biến thể của các câu này bị loại khỏi train)
    eval_sentences = eval_err_sents | eval_keep_sents | eval_syn_sents

    # ── 4) EVAL SETS ──
    eval_real_error = _std(real_err[real_err["ground_truth"].isin(eval_err_sents)], "real_error")

    eval_keep_correct = _std(
        real_ok[real_ok["ground_truth"].isin(eval_keep_sents)], "keep_correct"
    ).drop_duplicates(subset=["ground_truth"]).reset_index(drop=True)

    syn_eval_pool = syn[syn["ground_truth"].isin(eval_syn_sents)]
    eval_syn_rows = (
        syn_eval_pool.groupby("ground_truth", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), syn_eval_per_sentence), random_state=seed))
    )
    eval_synthetic = _std(eval_syn_rows, "synthetic").reset_index(drop=True)

    # ── 5) TRAIN SET (mọi thứ có câu KHÔNG thuộc eval_sentences) ──
    # 5a. real error (train) × weight
    real_err_train = real_err[~real_err["ground_truth"].isin(eval_sentences)]
    real_err_train_std = _std(real_err_train, "real_error")

    # 5b. keep-correct từ câu THẬT (identity gt→gt) × weight
    real_ok_train = real_ok[~real_ok["ground_truth"].isin(eval_sentences)].drop_duplicates(
        subset=["ground_truth"]
    )
    if keep_correct_train and len(real_ok_train) > keep_correct_train:
        real_ok_train = real_ok_train.sample(n=keep_correct_train, random_state=seed)
    keep_real = real_ok_train.copy()
    keep_real["predicted_text"] = keep_real["ground_truth"]   # identity
    keep_real_std = _std(keep_real, "keep_correct")

    train_parts = []
    for _ in range(max(1, real_weight)):
        train_parts.append(real_err_train_std)
        train_parts.append(keep_real_std)

    # 5c. synthetic (train) — loại mọi câu thuộc eval_sentences; dedup RIÊNG phần synthetic
    syn_train = syn[~syn["ground_truth"].isin(eval_sentences)]
    syn_train_std = _std(syn_train, "synthetic").drop_duplicates(
        subset=["predicted_text", "ground_truth"]
    )
    train_parts.append(syn_train_std)

    # 5d. clean→clean identity từ câu synthetic (train) để chống over-correction
    syn_train_sents = [s for s in syn_train["ground_truth"].unique().tolist()]
    if clean_identity_train and syn_train_sents:
        rng.shuffle(syn_train_sents)
        pick = syn_train_sents[:clean_identity_train]
        clean_id = pd.DataFrame({"predicted_text": pick, "ground_truth": pick})
        clean_id["is_real"] = False
        clean_id["severity"] = "unknown"
        clean_id["cp_type"] = "unknown"
        clean_id["kind"] = "keep_correct_clean"
        train_parts.append(clean_id[STD_COLS])

    # KHÔNG dedup toàn cục (sẽ nuốt mất nhân bản ×weight của real). Synthetic đã dedup riêng ở 5c.
    train = pd.concat(train_parts, ignore_index=True)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    # ── 6) DEV (chọn checkpoint / early-stop): gộp 3 eval, đã sentence-disjoint với train ──
    dev = pd.concat([eval_real_error, eval_keep_correct, eval_synthetic], ignore_index=True)

    # ── 7) GUARD: không câu eval nào lọt vào train ──
    train_sents = set(train["ground_truth"])
    leak = train_sents & eval_sentences
    assert not leak, f"RÒ RỈ: {len(leak)} câu eval có trong train!"

    # ── 8) LƯU ──
    paths = {
        "train": os.path.join(out_dir, "train.csv"),
        "dev": os.path.join(out_dir, "dev.csv"),
        "eval_real_error": os.path.join(out_dir, "eval_real_error.csv"),
        "eval_keep_correct": os.path.join(out_dir, "eval_keep_correct.csv"),
        "eval_synthetic": os.path.join(out_dir, "eval_synthetic.csv"),
    }
    train.to_csv(paths["train"], index=False, encoding="utf-8-sig")
    dev.to_csv(paths["dev"], index=False, encoding="utf-8-sig")
    eval_real_error.to_csv(paths["eval_real_error"], index=False, encoding="utf-8-sig")
    eval_keep_correct.to_csv(paths["eval_keep_correct"], index=False, encoding="utf-8-sig")
    eval_synthetic.to_csv(paths["eval_synthetic"], index=False, encoding="utf-8-sig")

    report = {
        "seed": seed,
        "real_total": int(len(real)),
        "real_error_pairs": int(len(real_err)),
        "real_error_sentences": int(len(err_sents)),
        "real_correct_rows": int(len(real_ok)),
        "synthetic_rows": int(len(syn)),
        "eval_sentences_held_out": int(len(eval_sentences)),
        "train_rows": int(len(train)),
        "train_breakdown": {k: int(v) for k, v in train["kind"].value_counts().to_dict().items()},
        "train_real_rows": int(train["is_real"].sum()),
        "train_real_error_vs_synthetic": (
            f"{int((train['kind'] == 'real_error').sum())} : "
            f"{int((train['kind'] == 'synthetic').sum())} "
            f"(1 : {int((train['kind'] == 'synthetic').sum()) / max(1, int((train['kind'] == 'real_error').sum())):.0f})"
        ),
        "dev_rows": int(len(dev)),
        "eval_real_error_rows": int(len(eval_real_error)),
        "eval_real_error_sentences": int(len(eval_err_sents)),
        "eval_keep_correct_rows": int(len(eval_keep_correct)),
        "eval_synthetic_rows": int(len(eval_synthetic)),
        "paths": paths,
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== BÁO CÁO DỰNG SPLIT CORRECTION (chống rò rỉ) ===")
    for k, v in report.items():
        if k not in ("paths", "train_breakdown"):
            print(f"  {k}: {v}")
    print(f"  train_breakdown: {report['train_breakdown']}")
    print(f"  -> {out_dir}/")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Dựng train/eval chống rò rỉ cho ViT5 corrector.")
    ap.add_argument("--real-csv", default="./outputs/correction_data_real.csv")
    ap.add_argument("--synthetic-csv", default="./outputs/correction_data_synthetic.csv")
    ap.add_argument("--out-dir", default="./outputs/correction_splits")
    ap.add_argument("--err-eval-ratio", type=float, default=0.30)
    ap.add_argument("--keep-correct-eval", type=int, default=300)
    ap.add_argument("--syn-eval", type=int, default=500)
    ap.add_argument("--syn-eval-per-sentence", type=int, default=2)
    ap.add_argument("--real-weight", type=int, default=3)
    ap.add_argument("--keep-correct-train", type=int, default=6000)
    ap.add_argument("--clean-identity-train", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_splits(
        real_csv=args.real_csv,
        synthetic_csv=args.synthetic_csv,
        out_dir=args.out_dir,
        err_eval_ratio=args.err_eval_ratio,
        keep_correct_eval=args.keep_correct_eval,
        syn_eval_sentences=args.syn_eval,
        syn_eval_per_sentence=args.syn_eval_per_sentence,
        real_weight=args.real_weight,
        keep_correct_train=args.keep_correct_train,
        clean_identity_train=args.clean_identity_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
