import gradio as gr
import time
import warnings
import os
import base64
import tempfile
import re
from inference import SpeechToTextPipeline

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

print("⏳ Đang tải toàn bộ hệ thống...")
pipeline = SpeechToTextPipeline(config_path="./config/config.yaml") 
print("✅ Tải mô hình hoàn tất!")

def process_audio(audio_path):
    if audio_path is None:
        return "⚠️ Vui lòng thu âm hoặc tải lên file audio.", "❌ Chưa có dữ liệu", "0.0s", None
    
    start_time = time.time()
    try:
        asr_text, gec_text, audio_output_path = pipeline.transcribe_and_correct(audio_path)
        if gec_text:
            gec_text = " ".join(gec_text.split())
            gec_text = re.sub(r'[^\w\s,.]', '', gec_text)
            gec_text = gec_text.strip()
        process_time = round(time.time() - start_time, 2)
        
        if asr_text is None:
             return "❌ Không thể xử lý âm thanh", "❌ Lỗi", "0.0s", None
             
        return asr_text, gec_text, f"{process_time}s", audio_output_path
        
    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}", "❌ Lỗi!", "0.0s", None

def process_base64_audio(base64_string):
    if not base64_string:
        return None
    try:
        header, encoded = base64_string.split(",", 1)
        data = base64.b64decode(encoded)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(data)
            temp_path = f.name
        return temp_path
    except Exception:
        return None

# Custom CSS cho giao diện chuyên nghiệp
custom_css = """
/* ===== GLOBAL STYLES ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #1e40af;
    --secondary-color: #3b82f6;
    --accent-color: #60a5fa;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #334155;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ===== HEADER ===== */
#header-title {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.3);
    border: 1px solid rgba(59, 130, 246, 0.2);
}

#header-title h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-align: center;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

#header-subtitle {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    text-align: center;
    margin-top: 0.5rem !important;
}

/* ===== METRICS BANNER ===== */
#metrics-container {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metric-card {
    background: rgba(59, 130, 246, 0.1);
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(59, 130, 246, 0.2);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    background: rgba(59, 130, 246, 0.15);
}

.metric-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}

.metric-label {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    margin-top: 0.5rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== CARDS & BOXES ===== */
.input-panel, .output-panel {
    background: var(--card-bg) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
    height: 100%;
}

.panel-header {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 1rem !important;
    border-bottom: 2px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

/* ===== BUTTONS ===== */
button {
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

.primary-btn {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
    color: white !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.4) !important;
}

#btn_start {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
    flex: 1;
    height: 60px !important;
    font-size: 1.1rem !important;
}

#btn_start:hover {
    box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4) !important;
}

#btn_stop {
    background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%) !important;
    flex: 1;
    height: 60px !important;
    font-size: 1.1rem !important;
}

#btn_run {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    width: 100% !important;
    height: 60px !important;
    font-size: 1.2rem !important;
    margin-top: 1.5rem !important;
}

#btn_run:hover {
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
}

/* ===== INPUT FIELDS ===== */
.gr-box, .gr-form, .gr-input {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.75rem !important;
}

textarea, input {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

textarea:focus, input:focus {
    border-color: var(--secondary-color) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* ===== AUDIO COMPONENTS ===== */
.gr-audio {
    background: rgba(15, 23, 42, 0.5) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    padding: 1rem !important;
}

/* ===== STEP INDICATORS ===== */
.step-container {
    background: rgba(30, 64, 175, 0.1);
    border-left: 4px solid var(--secondary-color);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.step-title {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--accent-color) !important;
    margin-bottom: 0.75rem !important;
}

/* ===== EXAMPLES SECTION ===== */
#examples-section {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 2rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.example-item {
    background: rgba(59, 130, 246, 0.05);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.1);
    transition: all 0.3s ease;
}

.example-item:hover {
    background: rgba(59, 130, 246, 0.1);
    border-color: rgba(59, 130, 246, 0.3);
}

/* ===== LOADING ANIMATION ===== */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    #header-title h1 {
        font-size: 1.75rem !important;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
    }
    
    .input-panel, .output-panel {
        padding: 1.5rem !important;
    }
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--secondary-color);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-color);
}
"""

def create_ui():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        # Header
        with gr.Group(elem_id="header-title"):
            gr.Markdown(
                """
                # 🎯 Trợ Lý Giao Tiếp AI - Hỗ Trợ Người Nói Ngọng
                ### Công nghệ ASR + GEC + TTS cho Dysarthria
                """,
                elem_id="header-subtitle"
            )
        
        # Metrics Banner
        with gr.Group(elem_id="metrics-container"):
            gr.Markdown("### 📊 Hiệu Năng Hệ Thống", elem_classes="panel-header")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-value">-1.41%</div>
                            <div class="metric-label">GAP Score</div>
                        </div>
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-value">87.57</div>
                            <div class="metric-label">BLEU Score</div>
                        </div>
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-value">2.58%</div>
                            <div class="metric-label">WER</div>
                        </div>
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-value">1.49%</div>
                            <div class="metric-label">CER</div>
                        </div>
                        """
                    )
        
        # Main Content - 2 Columns
        with gr.Row(equal_height=True):
            # LEFT COLUMN - INPUT
            with gr.Column(scale=1, elem_classes="input-panel"):
                gr.Markdown("## 🎤 Đầu Vào", elem_classes="panel-header")
                
                gr.Markdown(
                    """
                    <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #3b82f6;">
                        <strong>📌 Hướng dẫn sử dụng:</strong><br>
                        <strong>Bước 1:</strong> Nhấn nút "THU ÂM" và nói rõ ràng<br>
                        <strong>Bước 2:</strong> Nhấn "Dừng" khi hoàn tất<br>
                        <strong>Bước 3:</strong> Nhấn "PHIÊN DỊCH" để xử lý
                    </div>
                    """
                )
                
                # Recording Controls
                with gr.Row():
                    btn_start = gr.Button("🎙️ THU ÂM", variant="primary", elem_id="btn_start")
                    btn_stop = gr.Button("⏹️ Dừng", variant="secondary", interactive=False, elem_id="btn_stop")
                
                dummy_input = gr.Textbox(visible=False)
                
                audio_input = gr.Audio(
                    label="🎧 Nghe Lại Bản Ghi", 
                    type="filepath", 
                    interactive=False,
                    elem_classes="step-container"
                )
                
                btn_run = gr.Button("🚀 PHIÊN DỊCH", variant="primary", size="lg", elem_id="btn_run")
                
                out_time = gr.Textbox(
                    label="⏱️ Thời Gian Xử Lý", 
                    interactive=False,
                    elem_classes="step-container"
                )
            
            # RIGHT COLUMN - OUTPUT/ANALYSIS
            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.Markdown("## 📊 Kết Quả Phân Tích", elem_classes="panel-header")
                
                # Step 1: ASR
                gr.Markdown(
                    """
                    <div class="step-container">
                        <div class="step-title">🔍 BƯỚC 1: Nhận Dạng Giọng Nói (ASR)</div>
                    </div>
                    """
                )
                out_asr = gr.Textbox(
                    label="Văn Bản Thô", 
                    lines=3,
                    placeholder="Văn bản được nhận dạng từ giọng nói sẽ hiển thị tại đây...",
                    interactive=False
                )
                
                # Step 2: GEC
                gr.Markdown(
                    """
                    <div class="step-container">
                        <div class="step-title">✨ BƯỚC 2: Hiệu Chỉnh Ngữ Pháp (GEC)</div>
                    </div>
                    """
                )
                out_gec = gr.Textbox(
                    label="Văn Bản Đã Chỉnh Sửa", 
                    lines=3,
                    placeholder="Văn bản sau khi hiệu chỉnh ngữ pháp sẽ hiển thị tại đây...",
                    interactive=False
                )
                
                # Step 3: TTS
                gr.Markdown(
                    """
                    <div class="step-container">
                        <div class="step-title">🔊 BƯỚC 3: Tổng Hợp Giọng Nói (TTS)</div>
                    </div>
                    """
                )
                out_audio = gr.Audio(
                    label="Âm Thanh Đầu Ra", 
                    autoplay=True,
                    interactive=False
                )
        
        # Examples Section
        with gr.Group(elem_id="examples-section"):
            gr.Markdown("## 💡 Case Studies & Examples", elem_classes="panel-header")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="example-item">
                            <strong>📝 Ví dụ 1: Câu hỏi thông thường</strong><br>
                            <em>Input:</em> "Hôm nay ... thời tiết ... thế nào"<br>
                            <em>Output:</em> "Hôm nay thời tiết thế nào?"<br>
                            <span style="color: #10b981;">✓ Độ chính xác: 98.5%</span>
                        </div>
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="example-item">
                            <strong>📝 Ví dụ 2: Câu phức tạp</strong><br>
                            <em>Input:</em> "Tôi muốn ... đi ... bệnh viện"<br>
                            <em>Output:</em> "Tôi muốn đi bệnh viện."<br>
                            <span style="color: #10b981;">✓ Độ chính xác: 96.2%</span>
                        </div>
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        <div class="example-item">
                            <strong>📝 Ví dụ 3: Yêu cầu hỗ trợ</strong><br>
                            <em>Input:</em> "Giúp tôi ... với"<br>
                            <em>Output:</em> "Giúp tôi với."<br>
                            <span style="color: #10b981;">✓ Độ chính xác: 99.1%</span>
                        </div>
                        """
                    )
        
        # Footer
        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: rgba(30, 64, 175, 0.1); border-radius: 12px;">
                <p style="color: #94a3b8; margin: 0;">
                    🔬 Powered by Advanced AI Pipeline | ASR: PhoWhisper + LoRA | GEC: ViT5 | TTS: MMS-VITS
                </p>
            </div>
            """
        )
        
        # ===== JAVASCRIPT LOGIC (KHÔNG THAY ĐỔI) =====
        start_js = """
        async () => {
            try {
                window.my_stream = await navigator.mediaDevices.getUserMedia({
                    audio: { 
                        channelCount: 1, 
                        sampleRate: 16000, 
                        echoCancellation: false, 
                        noiseSuppression: false, 
                        autoGainControl: false,
                        googAutoGainControl: false,
                        googNoiseSuppression: false,
                        googHighpassFilter: false
                    }
                });
                window.my_mediaRecorder = new MediaRecorder(window.my_stream);
                window.my_audioChunks = [];
                window.my_mediaRecorder.ondataavailable = e => window.my_audioChunks.push(e.data);
                window.my_mediaRecorder.start();
            } catch (err) {
                alert("Lỗi Micro: Vui lòng kiểm tra quyền truy cập hoặc đảm bảo chạy trên môi trường HTTPS. " + err.message);
                throw err;
            }
        }
        """
        
        stop_js = """
        async (dummy) => {
            document.querySelector('#btn_stop').innerText = "⏳ Đang xử lý...";
            return new Promise((resolve) => {
                if(!window.my_mediaRecorder) { resolve(""); return; }
                
                window.my_mediaRecorder.onstop = async () => {
                    const blob = new Blob(window.my_audioChunks, { type: 'audio/webm' });
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const decodedAudio = await audioContext.decodeAudioData(arrayBuffer);
                    const offlineContext = new OfflineAudioContext(1, decodedAudio.duration * 16000, 16000);
                    const source = offlineContext.createBufferSource();
                    source.buffer = decodedAudio;
                    source.connect(offlineContext.destination);
                    source.start();
                    const resampledAudio = await offlineContext.startRendering();
                    
                    const rawData = resampledAudio.getChannelData(0);
                    let sumSquares = 0;
                    for (let i = 0; i < rawData.length; i++) sumSquares += rawData[i] * rawData[i];
                    const rms = Math.sqrt(sumSquares / rawData.length);
                    const currentDb = 20 * Math.log10(rms || 0.0001); 
                    const gain = Math.pow(10, (-20.0 - currentDb) / 20);
                    for (let i = 0; i < rawData.length; i++) {
                        rawData[i] *= gain;
                        if (rawData[i] > 1) rawData[i] = 0.99;
                        if (rawData[i] < -1) rawData[i] = -0.99;
                    }
                    
                    const numOfChan = resampledAudio.numberOfChannels;
                    const length = resampledAudio.length * numOfChan * 2 + 44;
                    const buffer = new ArrayBuffer(length);
                    const view = new DataView(buffer);
                    const channels = [];
                    let offset = 0, pos = 0;
                    function setUint16(data) { view.setUint16(pos, data, true); pos += 2; }
                    function setUint32(data) { view.setUint32(pos, data, true); pos += 4; }
                    setUint32(0x46464952); setUint32(length - 8); setUint32(0x45564157);
                    setUint32(0x20746d66); setUint32(16); setUint16(1); setUint16(numOfChan);
                    setUint32(16000); setUint32(16000 * 2 * numOfChan);
                    setUint16(numOfChan * 2); setUint16(16); setUint32(0x61746164); setUint32(length - pos - 4);
                    for (let i = 0; i < resampledAudio.numberOfChannels; i++) channels.push(resampledAudio.getChannelData(i));
                    while (pos < length) {
                        for (let i = 0; i < numOfChan; i++) {
                            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
                            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                            view.setInt16(pos, sample, true); pos += 2;
                        }
                        offset++;
                    }
                    
                    const wavBlob = new Blob([buffer], { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        resolve(reader.result);
                    };
                    reader.readAsDataURL(wavBlob);
                };
                
                setTimeout(() => {
                    window.my_mediaRecorder.stop();
                    if(window.my_stream) {
                        window.my_stream.getTracks().forEach(track => track.stop());
                    }
                }, 1000);
            });
        }
        """
        
        def on_start_click():
            return gr.update(value="🔴 Đang thu âm...", interactive=False), gr.update(interactive=True)
            
        def on_stop_click(b64_str):
            audio_path = process_base64_audio(b64_str)
            return (
                gr.update(value="🎙️ THU ÂM", interactive=True), 
                gr.update(value="⏹️ Dừng", interactive=False),
                audio_path
            )

        # Event Handlers
        btn_start.click(
            fn=on_start_click,
            outputs=[btn_start, btn_stop],
            js=start_js
        )
        
        btn_stop.click(
            fn=on_stop_click,
            inputs=[dummy_input],
            outputs=[btn_start, btn_stop, audio_input],
            js=stop_js
        )
        
        btn_run.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[out_asr, out_gec, out_time, out_audio] 
        )
        
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)