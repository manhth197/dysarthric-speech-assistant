"""
src/eval_correction.py — Đo ViT5 Corrector TRUNG THỰC trên 3 tập tách bạch.

Ba câu hỏi khác nhau, 3 tập khác nhau (dựng bởi build_correction_splits.py):
  eval_real_error   : cặp ASR-nghe-SAI thật (input != target)  → SỬA LỖI THẬT có tốt hơn ASR trần?
  eval_keep_correct : câu ASR nghe ĐÚNG (input == target)      → OVER-CORRECTION: model có phá câu đúng?
  eval_synthetic    : cặp synthetic (input != target)          → chỉ để tham chiếu, DÁN NHÃN "synthetic".

Mỗi tập đo 2 chế độ:
  raw   = output model thô (tắt Sanity Guard, min_similarity=-1)
  guard = có Sanity Guard (min_similarity=0.75, đúng như inference thật)

CÁCH DÙNG:
  # đo model hiện tại:
  python -m src.eval_correction --model ./outputs/vit5_correction_best --tag current
  # đo model train lại:
  python -m src.eval_correction --model ./outputs/vit5_correction_clean --tag clean
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import sacrebleu
import torch

from .model import ViT5Corrector
from .utils import normalize_vietnamese_text

norm = normalize_vietnamese_text


def _bleu(hyps, refs) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def _rate(pred, cond) -> float:
    if not pred:
        return 0.0
    return 100.0 * sum(cond) / len(pred)


def _run_set(corrector, csv_path, device, batch_size, num_beams):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    inputs = df["predicted_text"].astype(str).tolist()
    targets = df["ground_truth"].astype(str).tolist()

    raw_out = corrector.correct_batch(
        inputs, device=device, min_similarity=-1.0, num_beams=num_beams, batch_size=batch_size
    )
    guard_out = corrector.correct_batch(
        inputs, device=device, min_similarity=0.75, num_beams=num_beams, batch_size=batch_size
    )

    i_n = [norm(x) for x in inputs]
    t_n = [norm(x) for x in targets]
    r_n = [norm(x) for x in raw_out]
    g_n = [norm(x) for x in guard_out]

    n = len(inputs)
    is_identity = all(i_n[k] == t_n[k] for k in range(n)) if n else False

    m = {
        "n": n,
        "bleu_input_vs_target": round(_bleu(i_n, t_n), 2),
        "bleu_raw_vs_target": round(_bleu(r_n, t_n), 2),
        "bleu_guard_vs_target": round(_bleu(g_n, t_n), 2),
        "exact_match_guard_%": round(_rate(g_n, [g_n[k] == t_n[k] for k in range(n)]), 2),
    }
    # delta so với ASR trần (dương = corrector giúp; âm = corrector phá)
    m["delta_bleu_guard_minus_input"] = round(m["bleu_guard_vs_target"] - m["bleu_input_vs_target"], 2)

    if is_identity:
        # tập giữ-đúng: đo tỷ lệ model ĐỔI câu vốn đã đúng (over-correction)
        m["over_correction_raw_%"] = round(_rate(r_n, [r_n[k] != i_n[k] for k in range(n)]), 2)
        m["over_correction_guard_%"] = round(_rate(g_n, [g_n[k] != i_n[k] for k in range(n)]), 2)
    else:
        # tập có lỗi: đo tỷ lệ sửa THÀNH đúng, và tỷ lệ giữ nguyên (guard trả input)
        m["fixed_to_exact_guard_%"] = round(_rate(g_n, [g_n[k] == t_n[k] for k in range(n)]), 2)
        m["guard_kept_input_%"] = round(_rate(g_n, [g_n[k] == i_n[k] for k in range(n)]), 2)
    return m


def evaluate_model(
    model_path: str,
    splits_dir: str = "./outputs/correction_splits",
    out_dir: str = "./outputs/correction_eval",
    tag: str = "model",
    batch_size: int = 32,
    num_beams: int = 4,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrector = ViT5Corrector(model_name=model_path)

    sets = {
        "eval_real_error": os.path.join(splits_dir, "eval_real_error.csv"),
        "eval_keep_correct": os.path.join(splits_dir, "eval_keep_correct.csv"),
        "eval_synthetic": os.path.join(splits_dir, "eval_synthetic.csv"),
    }
    results = {"model": model_path, "tag": tag, "device": device}
    for name, path in sets.items():
        if not os.path.exists(path):
            print(f"⚠️  Bỏ qua {name}: không thấy {path}")
            continue
        print(f"▶ Đang đo {name} ...")
        results[name] = _run_set(corrector, path, device, batch_size, num_beams)

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"eval_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # In bảng gọn
    print("\n" + "=" * 66)
    print(f"KẾT QUẢ ĐO THẬT — model='{tag}' ({model_path})")
    print("=" * 66)
    a = results.get("eval_real_error", {})
    b = results.get("eval_keep_correct", {})
    c = results.get("eval_synthetic", {})
    if a:
        print(f"[A] SỬA LỖI THẬT (n={a['n']}):")
        print(f"    ASR trần   BLEU = {a['bleu_input_vs_target']}")
        print(f"    +Corrector BLEU = {a['bleu_guard_vs_target']}  (Δ {a['delta_bleu_guard_minus_input']:+})")
        print(f"    sửa-thành-đúng = {a.get('fixed_to_exact_guard_%')}% | giữ nguyên = {a.get('guard_kept_input_%')}%")
    if b:
        print(f"[B] GIỮ-ĐÚNG / over-correction (n={b['n']}):")
        print(f"    model ĐỔI câu đúng: raw = {b.get('over_correction_raw_%')}%  →  sau Guard = {b.get('over_correction_guard_%')}%")
        print(f"    BLEU giữ-đúng (guard) = {b['bleu_guard_vs_target']} (100 = hoàn hảo)")
    if c:
        print(f"[C] SYNTHETIC (n={c['n']}) — chỉ tham chiếu, KHÔNG phải hiệu năng thật:")
        print(f"    ASR trần BLEU = {c['bleu_input_vs_target']} | +Corrector BLEU = {c['bleu_guard_vs_target']} (Δ {c['delta_bleu_guard_minus_input']:+})")
    print("=" * 66)
    print(f"💾 {out_json}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo ViT5 corrector trung thực (3 tập tách bạch).")
    ap.add_argument("--model", default="./outputs/vit5_correction_best")
    ap.add_argument("--splits-dir", default="./outputs/correction_splits")
    ap.add_argument("--out-dir", default="./outputs/correction_eval")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=4)
    args = ap.parse_args()
    evaluate_model(
        model_path=args.model,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        tag=args.tag,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
