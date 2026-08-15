# AI4Life — Trợ lý giao tiếp cho người nói khó (dysarthria) tiếng Việt

> A 3-stage speech pipeline that turns hard-to-understand Vietnamese speech
> (cerebral palsy / dysarthria) into clear text and speech.
> Built for the disability community and hospitals — offline & privacy-first.

**Tiếng Việt:** AI4Life dịch giọng nói khó nghe của người bại não / nói ngọng
tiếng Việt thành **văn bản chuẩn + giọng đọc rõ ràng**, giúp họ giao tiếp. Dự án
nghiên cứu (NCKH, ĐH Bách khoa Hà Nội), do một lead developer mắc bại não phát triển.

---

## Pipeline

```
Giọng nói khó (dysarthria)
        │
        ▼
[1] ASR   — PhoWhisper-medium + LoRA        →  văn bản thô
        │
        ▼
[2] GEC   — ViT5 (post-ASR correction)      →  văn bản chuẩn
        │       + Sanity Guard (chống over-correction)
        ▼
[3] TTS   — MMS-VITS (facebook/mms-tts-vie)  →  giọng đọc rõ
```

- **ASR**: PhoWhisper-medium fine-tune bằng LoRA cho giọng dysarthria.
- **GEC**: ViT5 sửa lỗi hậu-ASR; có Sanity Guard (difflib) để không "sửa quá tay".
- **TTS**: MMS-VITS đọc lại câu đã chuẩn hoá.
- **Dữ liệu**: chiến lược "Mỏ neo & Cánh buồm" — dữ liệu thật + dữ liệu tổng hợp
  mô phỏng bệnh học (3 thể: spastic / athetoid / ataxic).

## Cấu trúc repo

```
config/           cấu hình pipeline (config.yaml)
src/              model, dataset, train, eval, sinh dữ liệu, metrics
inference.py      chạy full pipeline ASR → GEC → TTS trên 1 file audio
server.py         FastAPI backend cho app điện thoại (hybrid)
app/              app Flutter (Android) — client mỏng gọi server
demo.py           demo Gradio
requirements-win.txt / requirements.txt
```

> **Lưu ý:** model weights (`outputs/`), dữ liệu audio & giọng bệnh nhân (`dataset/`,
> `*_real.csv`) **không** nằm trong repo (bảo mật + dung lượng). Xem `.gitignore`.

## Chạy thử

```bash
python -m venv venv && venv/Scripts/activate        # Windows
pip install -r requirements-win.txt                  # (hoặc requirements.txt trên Linux)

# 1) Suy diễn 1 file:
python inference.py --audio test.wav

# 2) Backend cho app điện thoại:
python server.py            # FastAPI :8000  (đặt AI4LIFE_TOKEN để bật auth)

# 3) Demo web:
python demo.py
```

## Trạng thái

Dự án nghiên cứu, giai đoạn đầu, đang được dọn dẹp & mở nguồn. Đánh giá đang được
làm lại theo hướng chống rò rỉ (chia theo người nói / theo câu, tách real vs synthetic).

## License

TBD.
