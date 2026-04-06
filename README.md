# 🎙️ Dysarthric Speech Assistant: AI-Powered Recognition \& Contextual Correction Pipeline
## Built from my own lived experience with cerebral palsy and dysarthria, this project aims to turn AI into a practical communication bridge for people whose voices are often misunderstood.

> A research-driven assistive AI system designed to help people with dysarthria communicate more clearly by combining speech recognition, contextual correction, and speech synthesis in a real-time pipeline.

> **Note:** The source code for this project is currently private. This repository is intended to present the project motivation, system design, methodology, and results.

## 🚀 Project Overview

Traditional speech recognition systems such as **Google Speech** or base **Whisper** often perform poorly on dysarthric speech. Because dysarthric speech can contain irregular articulation, atypical phonemes, unstable rhythm, and inconsistent tone production, standard models frequently produce unusable transcriptions, sometimes with **Word Error Rates (WER) above 70%**.

To address this problem, I designed a **3-tier assistive pipeline**:

1. **Tier 1 – Automatic Speech Recognition (ASR)**  
   Extracts text from dysarthric speech audio.

2. **Tier 2 – Contextual Correction (GEC / Language Model)**  
   Repairs noisy transcription into fluent, context-aware Vietnamese text.

3. **Tier 3 – Text-to-Speech (TTS)**  
   Converts corrected text into clear synthesized speech, allowing the user to communicate more effectively with listeners.

This modular design separates the tasks of **hearing**, **understanding**, and **speaking**, which makes the system more robust, interpretable, and easier to improve over time.

## 👨‍💻 My Role

I independently led and developed the project from idea to implementation, including:

- Identifying the real-world problem from personal lived experience
- Designing the **3-tier AI architecture**
- Fine-tuning the **PhoWhisper** acoustic model for dysarthric Vietnamese speech
- Building the **ViT5-based contextual correction module**
- Designing the **TTS output stage** for clearer spoken communication
- Creating an automated **synthetic data generation pipeline**
- Applying **LoRA** for resource-efficient training
- Optimizing end-to-end inference for **real-time interaction**
- Evaluating system quality with **BLEU** and latency benchmarks

---

## 🧠 System Architecture

The assistant is structured as a sequential 3-tier pipeline.

### Tier 1 — “The Ears” (Acoustic Model / ASR)
This layer uses a fine-tuned **PhoWhisper** model to transcribe dysarthric Vietnamese speech into raw text.

Its goal is to focus on **acoustic recognition** rather than grammatical correctness. In other words, this stage tries to capture what was said as faithfully as possible from the audio signal, even if the output text is still noisy or incomplete.

### Tier 2 — “The Brain” (Contextual Correction / Language Model)
This layer uses **ViT5** in a text-to-text setup as a **Grammatical Error Correction (GEC)** and contextual reconstruction model.

It receives imperfect text from Tier 1 and transforms it into more fluent, coherent, and contextually correct Vietnamese. This is important because dysarthric speech recognition errors are often not random—they can be systematic and require contextual reasoning to recover the intended meaning.

### Tier 3 — “The Voice” (Speech Synthesis / TTS)
This final layer converts the corrected text into **clear synthesized Vietnamese speech**.

The purpose of this stage is not only to display corrected text on screen, but also to create a practical assistive communication loop:  
the user speaks → the system understands and corrects → the system speaks back clearly to other people.

This makes the solution more useful in real-life interactions such as conversations, school, work, medical settings, or public communication.

### End-to-End Flow

`Audio Input → PhoWhisper ASR → Raw Transcription → ViT5 Correction → Corrected Text → TTS Output → Clear Speech`

---


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
##### │   ├── train_asr.py     # Fine-tuning script for Tier 1 (PhoWhisper)

##### │   ├── train_correction.py     # Fine-tuning script for Tier 2 (ViT5)

##### │   └── generate_correction\_data.py # Data pipeline for simulating medical speech errors

##### ├── inference.py            # Integration class for the 2-Tier Pipeline

##### └── app_demo.py             # Gradio web interface for real-time interaction

## 💡 Key Technical Contributions

### 1. Dysarthric Speech Recognition with Domain Adaptation
Standard ASR models struggle with dysarthric speech because they are trained mostly on standard pronunciation. To improve performance, I fine-tuned **PhoWhisper** specifically for dysarthric Vietnamese speech patterns.

### 2. Preventing Catastrophic Forgetting with Data Mixing
**Challenge:** Fine-tuning only on dysarthric speech made the model overfit to narrow conditions and reduced its general robustness.

**Solution:** I used a **data mixing strategy**:

- **80% dysarthric speech data**
- **20% clean Vietnamese speech** from the **VIVOS corpus**

This helped the model adapt to atypical speech while preserving its general Vietnamese recognition ability.

### 3. Parameter-Efficient Fine-Tuning with LoRA
To reduce training cost, I applied **LoRA (Low-Rank Adaptation)** to a **1.5B-parameter PhoWhisper model**, updating only:

- **~24.5K trainable parameters**
- **0.0016% of the total model**

This significantly lowered VRAM and compute requirements while retaining the base model’s language knowledge.

### 4. Synthetic Data Generation for Tier 2 Training
Paired correction data for dysarthric Vietnamese speech is scarce. To overcome this, I built an automated pipeline that simulates clinical speech characteristics across three dysarthria-related patterns:

- **Spastic**
- **Athetoid**
- **Ataxic**

By injecting phonetic and tonal distortions into clean Vietnamese text, I generated **300,000+ synthetic sentence pairs** to train the contextual correction model.

### 5. Real-Time Inference Optimization
To improve responsiveness, I merged LoRA adapters into the base model and optimized the inference flow for practical use.

- **ASR + correction latency:** ~**0.2s per sentence**
- **Full 3-tier interaction:** designed for **near real-time usage**, depending on the TTS backend

### 6. Accessibility-First Design
This system was not designed only as an academic experiment. It was designed as an **assistive communication tool**. The architecture prioritizes:

- real-time responsiveness
- clarity of final output
- modular upgrades
- future personalization for different speech profiles

---

## 📊 Performance Metrics

| Metric | Result |
|--------|--------|
| Validation BLEU Score (Tier 2 - ViT5) | **95.81** |
| ASR + Correction Inference Time | **~0.5s / sentence** |
| Baseline Dysarthric Speech Recognition | **WER often > 90%** |
| Synthetic Correction Dataset Size | **300,000+ sentence pairs** |

These results show that the proposed architecture can significantly improve the usability of dysarthric Vietnamese speech processing while remaining practical for interactive communication support.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Model Libraries:** Hugging Face Transformers, PEFT
- **ASR:** PhoWhisper
- **Correction Model:** ViT5
- **Optimization:** LoRA, data mixing, adapter merging
- **Demo Interface:** Gradio
- **Speech Output:** Vietnamese TTS module

---

## 🎯 Why This Project Matters

For many people with cerebral palsy, speech is not a lack of thought—it is a mismatch between intention and how the voice is physically produced.

This project aims to reduce that barrier.

My goal is to build AI that does not simply measure speech against “normal” pronunciation, but instead learns to **understand the speaker**, preserve meaning, and help communicate it clearly. In that sense, this project is about more than transcription accuracy; it is about **accessibility, dignity, and communication independence**.

---

## 🔭 Future Vision

This project began as a personal solution, but the long-term goal is to develop it into a scalable assistive platform for **many people with cerebral palsy and dysarthria**.

Planned future directions include:

- Building **personalized speaker profiles** for different users
- Supporting multiple **severity levels** and speech patterns
- Improving adaptation to more real-world recording conditions
- Expanding the dataset with more diverse dysarthric Vietnamese speech
- Packaging the system into a **web/mobile assistive application**
- Exploring **personalized TTS voices** or safe voice-banking options
- Collaborating with families, therapists, and disability communities
- Turning the project into a practical communication tool for broader social use

---


# 🤝 Acknowledgments

##### This project is deeply inspired by the extraordinary efforts of the motor-impaired community in Vietnam. We also extend our gratitude to the developers of PhoWhisper (VinAI) and ViT5 for providing outstanding foundational Vietnamese models.

