"""
src/eval_split.py — Chia train/eval CHỐNG RÒ RỈ (leakage-safe) cho dự án voice.

VÌ SAO CẦN: cách cũ (`_split_data` trong train_correction.py) shuffle ngẫu nhiên theo TỪNG MẪU.
Hệ quả: các biến thể synthetic của cùng một câu gốc — hoặc cùng một câu người thật đọc nhiều lần —
rơi vào CẢ train lẫn eval. Model đã "thấy" câu eval => BLEU/WER trông đẹp hơn khả năng tổng quát
hóa thật sự trên người/câu chưa từng gặp.

HÀM NÀY chia theo NHÓM (group) với các bảo đảm:
  1. Mỗi giá trị `group_col` chỉ nằm ở MỘT phía (train HOẶC eval), không bao giờ cả hai.
       - Correction: dùng group_col="ground_truth"  -> mọi biến thể của một câu ở cùng một phía.
       - ASR:        dùng group_col="<cột speaker>" -> mọi câu của một người ở cùng một phía.
  2. Eval mặc định CHỈ gồm dữ liệu THẬT (is_real=True) — báo cáo trên người/câu thật chưa train.
  3. Biến thể synthetic của các group được chọn làm eval bị LOẠI HẲN (không cho vào train => không rò).
  4. Guard cuối: không câu nào (text_col) xuất hiện ở cả train lẫn eval (bắt cả trường hợp 2 người
     đọc trùng câu trong ASR).
  5. Có assert kiểm tra 0 rò rỉ; in báo cáo rõ ràng.

CÁCH DÙNG (chạy trên máy có dữ liệu — thường là server):
  # Correction — chống rò biến thể synthetic cùng câu:
  python -m src.eval_split --csv outputs/correction_data_combined.csv \
         --group-col ground_truth --out-dir outputs/split_correction

  # ASR — chia theo người nói (cần cột speaker trong metadata.csv; đổi tên cột cho đúng):
  python -m src.eval_split --csv dataset/metadata.csv \
         --group-col speaker --text-col transcript --out-dir outputs/split_asr

Hoặc gọi trực tiếp trong code:
  from src.eval_split import grouped_eval_split
  train_df, eval_df, report = grouped_eval_split(df, group_col="ground_truth")
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    """Chuẩn hóa cột real/synthetic về bool (chịu được True/'True'/'true'/1/'1'/'yes')."""
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))


def grouped_eval_split(
    df: pd.DataFrame,
    group_col: str,
    eval_ratio: float = 0.1,
    real_col: Optional[str] = "is_real",
    text_col: Optional[str] = "ground_truth",
    real_only_eval: bool = True,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Trả về (train_df, eval_df, report).

    Tham số:
      group_col      : cột định danh nhóm — KHÔNG nhóm nào nằm cả train lẫn eval.
      eval_ratio     : tỷ lệ NHÓM (không phải mẫu) đưa vào eval.
      real_col       : cột phân biệt real/synthetic. None nếu dữ liệu toàn thật.
      text_col       : cột câu để guard trùng lặp train/eval. None để bỏ guard.
      real_only_eval : True -> eval chỉ giữ dòng thật; synthetic của nhóm-eval bị loại hẳn.
      stratify_col   : giữ tỷ lệ theo nhãn (vd 'cp_type') khi chọn nhóm-eval (theo nhãn đa số của nhóm).
      seed           : cố định để tái lập.
    """
    if group_col not in df.columns:
        raise ValueError(
            f"Không thấy cột group_col='{group_col}'. Cột hiện có: {list(df.columns)}"
        )

    df = df.copy()
    if real_col and real_col in df.columns:
        df["_is_real"] = _as_bool(df[real_col])
    else:
        df["_is_real"] = True  # coi như toàn dữ liệu thật

    rng = np.random.default_rng(seed)

    # Nhóm ứng viên cho eval: nhóm có ít nhất 1 dòng thật (nếu real_only_eval)
    if real_only_eval:
        candidate_groups = df.loc[df["_is_real"], group_col].dropna().unique().tolist()
    else:
        candidate_groups = df[group_col].dropna().unique().tolist()

    # Chọn nhóm-eval (stratified theo nhãn đa số của nhóm nếu có stratify_col)
    eval_groups: set = set()
    if stratify_col and stratify_col in df.columns:
        cand = df[df[group_col].isin(candidate_groups)]
        grp_label = cand.groupby(group_col)[stratify_col].agg(
            lambda s: s.value_counts().index[0]
        )
        for _, groups_in_label in grp_label.groupby(grp_label):
            g = list(groups_in_label.index)
            rng.shuffle(g)
            k = int(round(len(g) * eval_ratio))
            eval_groups.update(g[:k])
    else:
        g = list(candidate_groups)
        rng.shuffle(g)
        k = int(round(len(g) * eval_ratio))
        eval_groups.update(g[:k])

    in_eval_group = df[group_col].isin(eval_groups)

    if real_only_eval:
        eval_df = df[in_eval_group & df["_is_real"]].copy()
        n_dropped_synth = int((in_eval_group & ~df["_is_real"]).sum())  # synthetic của nhóm-eval -> loại
        train_df = df[~in_eval_group].copy()
    else:
        eval_df = df[in_eval_group].copy()
        n_dropped_synth = 0
        train_df = df[~in_eval_group].copy()

    # Guard cuối: bỏ khỏi eval mọi câu đã có trong train (bắt trùng câu chéo nhóm, vd ASR)
    n_sentence_guard = 0
    if text_col and text_col in df.columns:
        train_texts = set(train_df[text_col].astype(str))
        overlap = eval_df[text_col].astype(str).isin(train_texts)
        n_sentence_guard = int(overlap.sum())
        eval_df = eval_df[~overlap].copy()

    # Báo cáo (tính trước khi bỏ cột phụ)
    report = {
        "n_total_input": int(len(df)),
        "n_train": int(len(train_df)),
        "n_eval": int(len(eval_df)),
        "eval_toan_du_lieu_that": bool(eval_df["_is_real"].all()) if len(eval_df) else True,
        "so_nhom_tong": int(df[group_col].nunique()),
        "so_nhom_eval": int(len(eval_groups)),
        "loai_synthetic_cua_nhom_eval": n_dropped_synth,
        "loai_boi_guard_trung_cau": n_sentence_guard,
        "train_that": int(train_df["_is_real"].sum()),
        "train_synthetic": int((~train_df["_is_real"]).sum()),
    }

    # Kiểm tra 0 rò rỉ
    tr_groups = set(train_df[group_col].dropna())
    ev_groups = set(eval_df[group_col].dropna())
    assert tr_groups.isdisjoint(ev_groups), "RÒ RỈ: có group ở cả train lẫn eval!"
    if text_col and text_col in df.columns and len(eval_df):
        assert set(train_df[text_col].astype(str)).isdisjoint(
            set(eval_df[text_col].astype(str))
        ), "RÒ RỈ: có câu ở cả train lẫn eval!"

    for d in (train_df, eval_df):
        d.drop(columns=["_is_real"], inplace=True, errors="ignore")

    return train_df, eval_df, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Chia train/eval chống rò rỉ (theo speaker hoặc theo câu).")
    ap.add_argument("--csv", required=True, help="File CSV đầu vào")
    ap.add_argument("--group-col", required=True, help="Cột nhóm: 'ground_truth' (correction) hoặc cột speaker (ASR)")
    ap.add_argument("--text-col", default="ground_truth", help="Cột câu để guard trùng lặp (mặc định ground_truth)")
    ap.add_argument("--real-col", default="is_real", help="Cột phân biệt real/synthetic")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    ap.add_argument("--stratify-col", default=None, help="Vd 'cp_type' để giữ tỷ lệ theo thể bệnh")
    ap.add_argument("--no-real-only", action="store_true", help="Cho phép synthetic vào eval (KHÔNG khuyến nghị)")
    ap.add_argument("--out-dir", default="outputs/split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    train_df, eval_df, report = grouped_eval_split(
        df,
        group_col=args.group_col,
        eval_ratio=args.eval_ratio,
        real_col=args.real_col if args.real_col in df.columns else None,
        text_col=args.text_col if args.text_col in df.columns else None,
        real_only_eval=not args.no_real_only,
        stratify_col=args.stratify_col,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.csv")
    eval_path = os.path.join(args.out_dir, "eval.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")

    lines = ["=== BÁO CÁO CHIA TẬP (chống rò rỉ) ==="]
    lines += [f"  {k}: {v}" for k, v in report.items()]
    lines += [f"  -> train: {train_path}", f"  -> eval : {eval_path}"]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "split_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
