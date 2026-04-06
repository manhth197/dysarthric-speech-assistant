# 🎙️ Dysarthric Speech Assistant: AI-Powered Recognition \& Contextual Correction Pipeline
## Built from my own lived experience with cerebral palsy and dysarthria, this project aims to turn AI into a practical communication bridge for people whose voices are often misunderstood.

> A research-driven assistive AI system designed to help people with dysarthria communicate more clearly by combining speech recognition, contextual correction, and speech synthesis in a real-time pipeline.

> **Note:** The source code for this project is currently private. This repository is intended to present the project motivation, system design, methodology, and results.


# 📌 About The Project
<img width="1676" height="580" alt="demo" src="https://github.com/user-attachments/assets/3165fa03-046e-4414-962c-fe38aecdee0d" />

##### This scientific research project, addresses a significant communication challenge: building an robust AI system to recognize and correct speech from individuals with Dysarthria, primarily caused by Cerebral Palsy.

##### 

##### Traditional AI models (e.g., Google Speech, base Whisper) often fail, yielding Word Error Rates (WER) >70% when processing dysarthric speech due to atypical phonemes and unpredictable rhythms. To solve this, our project proposes a 2-Tier Pipeline Architecture combining an Acoustic Model and a Language Model, achieving high accuracy with real-time processing capabilities.

# 

# 🚀 System Architecture

The system employs a sequential 3-tier pipeline to create a complete end-to-end communication loop:

* **Tier 1 - "The Ears" (Acoustic Model):** Utilizes a fine-tuned **PhoWhisper** model. This tier focuses solely on extracting raw phonemes from the audio signal, prioritizing acoustic recognition over grammatical logic.
* **Tier 2 - "The Brain" (Language Model):** Employs **ViT5** (Text-to-Text). Acting as a Grammatical Error Correction (GEC) post-processor, it receives the raw, potentially flawed text from Tier 1, infers the context, and corrects it into fluent Vietnamese text.
* **Tier 3 - "The Mouth" (Speech Synthesizer):** Integrates a **Vietnamese Text-to-Speech (TTS)** engine. It takes the contextually corrected text from Tier 2 and synthesizes it into clear, natural-sounding audio, completing the communication process for the listener with minimal latency.
<img width="1919" height="738" alt="Screenshot 2026-03-18 135436" src="https://github.com/user-attachments/assets/5925defd-3b82-49f9-8548-f2c5c7486155" />

# 

# 💡 Key Technical Achievements

##### Combating "Catastrophic Forgetting" via Data Mixing:

##### 

##### Challenge: Fine-tuning solely on dysarthric data caused the model to overfit to specific recording environments and forget how to process standard speech.

##### 

##### Solution: Implemented a robust data mixing strategy, combining 80% dysarthric data with 20% clean Vietnamese data (from the VIVOS corpus) to act as a regularization "memory anchor."

##### 

##### Resource Optimization with LoRA (Low-Rank Adaptation):

##### 

##### Fine-tuned the PhoWhisper model (1.5B parameters) by updating only \~24.5K training parameters (0.0016%). This drastically reduced VRAM requirements while preserving the foundational Vietnamese language knowledge.

##### **End-to-End Voice Communication Loop:**
##### Upgraded the system from a simple transcription tool to a full assistive communication device by integrating a low-latency TTS module, allowing motor-impaired users to effectively and clearly "speak" to others in real-time.

# Automated Synthetic Data Generation Pipeline:

# 

##### Modeled the clinical speech characteristics of three Dysarthria types (Spastic, Athetoid, Ataxic). Engineered a script to automatically inject targeted phonetic and tonal noise into clean data, generating over 300,000 synthetic sentence pairs to train the Tier 2 (ViT5) model without manual recording.

##### 

##### Real-Time Inference Optimization:

##### 

##### Merged the LoRA weights into the Base Model. Optimized the sequential inference pipeline to achieve a total end-to-end response time of just \~0.2 seconds.

# 

# 📊 Performance Metrics

##### BLEU Score (Tier 2 - ViT5): Achieved 95.81 on the Validation set.

##### 

##### Inference Time: \~0.2s per sentence (Ensuring smooth, real-time communication).

##### 

##### Detailed Training/Validation Loss charts are available in the /docs directory.

# 

# 📁 Repository Structure

# Plaintext

# 

##### ├── config/          # Configuration files (.yaml) for ASR and ViT5

##### ├── dataset/         # Scripts for downloading and mixing data (VIVOS)

##### ├── src/             # Core source code

##### │   ├── model.py     # Model definitions (ViT5Corrector, WhisperLoRA)

##### │   ├── utils.py     # Text normalization and preprocessing utilities

##### │   ├── dataset.py   # DataLoaders and Log-Mel Spectrogram processing
##### │   ├── tts_module.py       # Text-to-Speech integration and audio playback
##### │   ├── train\_asr.py     # Fine-tuning script for Tier 1 (PhoWhisper)

##### │   ├── train\_correction.py     # Fine-tuning script for Tier 2 (ViT5)

##### │   └── generate\_correction\_data.py # Data pipeline for simulating medical speech errors

##### ├── inference.py            # Integration class for the 2-Tier Pipeline

##### └── app\_demo.py             # Gradio web interface for real-time interaction

# 🤝 Acknowledgments

##### This project is deeply inspired by the extraordinary efforts of the motor-impaired community in Vietnam. We also extend our gratitude to the developers of PhoWhisper (VinAI) and ViT5 for providing outstanding foundational Vietnamese models.

