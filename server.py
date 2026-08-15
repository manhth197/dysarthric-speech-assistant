r"""
server.py — Backend API cho app điện thoại (kiến trúc hybrid).
Chạy pipeline ASR(medium+LoRA) -> GEC(ViT5) -> TTS(MMS) trên PC này,
app Flutter gọi qua mạng. Dùng lại y nguyên SpeechToTextPipeline trong inference.py.

Chạy:  D:\voice\venv\Scripts\python.exe server.py   (hoặc bấm run_server.bat)
API:
  GET  /health           -> {"status":"ok"}
  POST /process (file)    -> {asr_text, gec_text, audio_base64(wav), process_time}
"""
import os, re, base64, tempfile, time, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference import SpeechToTextPipeline

# --- Cấu hình an toàn (chỉnh qua biến môi trường nếu cần) ---
MAX_UPLOAD_MB = int(os.environ.get("AI4LIFE_MAX_UPLOAD_MB", "20"))
# Nếu đặt AI4LIFE_TOKEN, app phải gửi header X-Auth-Token khớp thì mới được xử lý.
# Để trống = mở (như hiện tại). NÊN đặt token khi expose qua Cloudflare tunnel ra Internet.
AUTH_TOKEN = os.environ.get("AI4LIFE_TOKEN", "")

print("⏳ Đang tải mô hình (medium + ViT5 + TTS)... (lần đầu hơi lâu)")
pipeline = SpeechToTextPipeline(config_path="./config/config.yaml")
print("✅ Backend sẵn sàng!")

app = FastAPI(title="AI4Life Speech API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": pipeline.device}


@app.post("/process")
async def process(audio: UploadFile = File(...), x_auth_token: str | None = Header(default=None)):
    t0 = time.time()
    # Xác thực (chỉ bật khi AI4LIFE_TOKEN được đặt)
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        return JSONResponse({"error": "Không có quyền (token sai)"}, status_code=401)

    suffix = os.path.splitext(audio.filename or "a.wav")[1] or ".wav"
    data = await audio.read()
    if not data:
        return JSONResponse({"error": "File rỗng"}, status_code=400)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"error": f"File quá lớn (tối đa {MAX_UPLOAD_MB}MB)"}, status_code=413)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    out_audio_path = None
    try:
        result = pipeline.transcribe_and_correct(path)
        if not result or result[0] is None:
            return JSONResponse({"error": "Không xử lý được âm thanh"}, status_code=422)
        asr_text, gec_text, out_audio_path = result

        if gec_text:
            gec_text = " ".join(gec_text.split())
            gec_text = re.sub(r"[^\w\s,.]", "", gec_text).strip()

        audio_b64 = None
        if out_audio_path and os.path.exists(out_audio_path):
            with open(out_audio_path, "rb") as af:
                audio_b64 = base64.b64encode(af.read()).decode("ascii")

        return JSONResponse({
            "asr_text": asr_text,
            "gec_text": gec_text,
            "audio_base64": audio_b64,
            "process_time": round(time.time() - t0, 2),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Xóa audio sau xử lý (quyền riêng tư): cả file gửi lên lẫn file giọng nói TTS
        for p in (path, out_audio_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
