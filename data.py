import os
import pandas as pd
import gradio as gr
from inference import SpeechToTextPipeline

print("🚀 Đang khởi động hệ thống nhận diện...")
pipeline = SpeechToTextPipeline(config_path="./config/config.yaml")

CSV_PATH = "./outputs/correction_data_real.csv"

# ==========================================
# STATE: danh sách file và con trỏ hiện tại
# ==========================================
audio_files = []
current_index = 0

def load_folder(folder_path, start_index):
    """Quét folder, lấy toàn bộ file audio, set con trỏ về vị trí người dùng chọn và tự động nhận diện."""
    global audio_files, current_index

    if not folder_path or not os.path.isdir(folder_path):
        return "⚠️ Đường dẫn folder không hợp lệ!", None, "", ""

    SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    audio_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])
    
    if not audio_files:
        return "⚠️ Không tìm thấy file audio nào trong folder!", None, "", ""

    # Xử lý index: Giao diện hiển thị từ 1, nhưng code Python chạy từ 0
    start_idx = int(start_index) - 1
    current_index = max(0, min(start_idx, len(audio_files) - 1))

    first_file = audio_files[current_index]
    status = f"📂 Tìm thấy {len(audio_files)} file. Đang xử lý file {current_index + 1}/{len(audio_files)}: {os.path.basename(first_file)}"
    
    # TỰ ĐỘNG NHẬN DIỆN FILE ĐẦU TIÊN
    print(f"🎙️ Đang tự động nhận diện file đầu tiên: {first_file}")
    asr_text, _ = transcribe_current(first_file)
    
    return status, first_file, asr_text, asr_text


def transcribe_current(audio_path):
    """Nhận diện file hiện tại."""
    if not audio_path:
        return "⚠️ Chưa có file nào được chọn!", ""
    try:
        asr_text, _gec, _audio = pipeline.transcribe_and_correct(audio_path)
        return asr_text, asr_text  # điền sẵn vào cả ô ASR lẫn Ground Truth
    except Exception as e:
        return f"Lỗi: {str(e)}", ""


def save_and_next(audio_path, asr_text, correct_text):
    """Lưu correction rồi tự động chuyển sang file tiếp theo & tự động nhận diện."""
    global current_index

    # --- Lưu dữ liệu ---
    save_msg = ""
    if not asr_text or not correct_text:
        save_msg = "⚠️ Bỏ qua lưu: thiếu ASR hoặc Ground Truth."
    elif asr_text.strip().lower() == correct_text.strip().lower():
        save_msg = "ℹ️ Whisper đã nghe đúng, bỏ qua không lưu."
    else:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        new_data = pd.DataFrame([{
            "audio_path": audio_path,
            "predicted_text": asr_text,
            "ground_truth": correct_text,
            "is_real": True
        }])
        write_header = not os.path.exists(CSV_PATH)
        new_data.to_csv(CSV_PATH, mode='a', header=write_header, index=False, encoding='utf-8-sig')
        save_msg = f"✅ Đã lưu: [{asr_text}] → [{correct_text}]"

    # --- Chuyển file tiếp theo ---
    current_index += 1
    if current_index >= len(audio_files):
        return "🎉 Đã xử lý hết tất cả file trong folder!", None, "", ""

    next_file = audio_files[current_index]
    status = f"{save_msg}\n➡️ Đang tự động đọc file {current_index + 1}/{len(audio_files)}: {os.path.basename(next_file)}..."
    
    # TỰ ĐỘNG NHẬN DIỆN FILE TIẾP THEO
    print(f"🎙️ Đang tự động nhận diện file tiếp theo: {next_file}")
    next_asr_text, _ = transcribe_current(next_file)

    return status, next_file, next_asr_text, next_asr_text


# ==========================================
# GIAO DIỆN
# ==========================================
with gr.Blocks(title="Kỹ Sư Nói Ngọng - Batch Collector") as app:
    gr.Markdown("# 🎙️ Trạm Thu Thập Dữ Liệu - Batch Mode (Tự Động 100%)")
    gr.Markdown("Nhập đường dẫn và vị trí bắt đầu. Hệ thống sẽ tự động nghe, bạn chỉ việc sửa chữ sai và bấm Next.")

    with gr.Row():
        folder_input = gr.Textbox(label="📁 Đường dẫn folder audio", placeholder="/path/to/your/audio/folder")
        start_index_input = gr.Number(label="Vị trí bắt đầu (Từ file số mấy?)", value=1, precision=0)
        btn_load = gr.Button("📂 Load Folder & Nghe File Đầu", variant="primary")

    status_output = gr.Textbox(label="Trạng thái hệ thống", interactive=False, lines=2)

    with gr.Row():
        with gr.Column():
            audio_preview = gr.Audio(type="filepath", label="🔊 File đang xử lý", interactive=False)
            btn_transcribe = gr.Button("🤖 Nhận Diện Lại (Dùng khi bị lỗi mạng/treo)", variant="secondary")

        with gr.Column():
            text_asr = gr.Textbox(label="Kết quả ASR (Whisper)", interactive=False, lines=2)
            text_correct = gr.Textbox(label="Ground Truth (Sửa lỗi sai vào đây)", lines=2)
            btn_save_next = gr.Button("💾 Lưu & Tự động chạy file tiếp theo →", variant="stop")

    # --- Sự kiện ---
    btn_load.click(
        fn=load_folder,
        inputs=[folder_input, start_index_input],
        outputs=[status_output, audio_preview, text_asr, text_correct]
    )
    btn_transcribe.click(
        fn=transcribe_current,
        inputs=audio_preview,
        outputs=[text_asr, text_correct]
    )
    btn_save_next.click(
        fn=save_and_next,
        inputs=[audio_preview, text_asr, text_correct],
        outputs=[status_output, audio_preview, text_asr, text_correct]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)