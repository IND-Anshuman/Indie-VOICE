# Indie-VOICES: Multilingual Audio Generation for Indian Languages 🇮🇳

**Indie-VOICES** is a multilingual **text-to-speech (TTS)** system designed to generate expressive, human-like voices across 22+ major Indian languages.  
It combines **Indic-Parler-TTS** for speech synthesis with **Google Gemini** for context-aware, expressive voice tone generation — enabling dynamic, natural-sounding output suitable for podcasts, assistants, lectures, and more.

---

## 🌐 Overview

This project integrates **Gemini LLM** (for intelligent voice characterization) and **Indic-Parler-TTS** (for multi-lingual audio synthesis).  
It’s built to handle long-form inputs by chunking text, generating expressive tone descriptions, and synthesizing seamless voice audio batches.

### Key Features

- 🔹 Supports 22+ Indian languages (Hindi, Tamil, Bengali, Telugu, Marathi, Kannada, Malayalam, etc.)  
- 🔹 Integrates **Gemini LLM** to expand short voice cues into expressive, natural tone prompts  
- 🔹 Deterministic decoding for stable, reproducible results  
- 🔹 Batch-based long-text generation with silence merging between segments  
- 🔹 RMS normalization and anti-clipping for studio-quality output  
- 🔹 Built-in text preprocessing for sentence-balanced chunking  

---

## 🧩 System Architecture


### 1. **Gemini Voice Description Generator**
- Takes a short cue like *"friendly Hindi female teacher"*  
- Uses Gemini to expand it into a detailed tone prompt specifying emotion, pitch, pace, etc.  
- This description conditions the TTS model for expressive output.

### 2. **Indic-Parler-TTS Synthesis Engine**
- Tokenizes both **voice descriptor** and **text input**  
- Processes text in batches using deterministic decoding (no random sampling)  
- Normalizes and merges generated audio chunks into a single `.wav` file

---

## ⚙️ Installation Guide

Clone the repository and set up dependencies:

git clone https://github.com/your-username/Indie-VOICES.git
cd Indie-VOICES
pip install -r requirements.txt


**Recommended Python version:** 3.9+  
**Supported OS:** Linux / Windows / macOS  
**GPU:** Optional (CUDA-supported GPU for faster synthesis)

---

## 🔐 Environment Setup

Create a `.env` file for your API keys.  
Use `.env.example` as a template and rename it:


---

### 🚀 Setting Up Authentication

#### 1. **Gemini API Key**
Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/).  
Once you have it, set it in your terminal or `.env` file:


#### 2. **Hugging Face Authentication**

Login to Hugging Face using your access token to enable model downloads:
In terminal write : hf auth login

When prompted, paste your **Hugging Face token** from  
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🧠 How It Works

### Step 1: Initialization  
The class `IndicTTSLLM` initializes:
- Gemini client for natural tone expansion  
- Parler-TTS model and tokenizer (CPU/GPU auto-detection)  

### Step 2: Voice Cue Interpretation  
`generate_voice_description()` sends the cue to Gemini and returns a refined professional-style voice prompt.  
Example input:  
> “energetic Hindi female teacher voice”  
Output:  
> “Energetic and expressive Hindi female educator voice with natural intonation, moderate pace, and confident projection.”

### Step 3: Text Processing  
Text is split using regex into balanced, sentence-sized segments for smoother generation.

### Step 4: Audio Generation  
For each text batch:
- Tokens are prepared for description and text  
- Parler-TTS generates batch-wise audio  
- Normalization and anti-clipping applied  
- All batches merged with 150ms natural silence  

### Step 5: Saving Output  
Final speech combined and saved to WAV format at a specified path (e.g., `output.wav`).

---

## 🧪 Example Usage
from indie_voices import IndicTTSLLM

tts = IndicTTSLLM(batch_size=2)

text = """नमस्कार विद्यार्थियों, आज हम स्मार्टफोन के प्रभाव पर चर्चा करेंगे..."""

voice_style = """Generate a clear, engaging male Hindi teacher voice for an educational lecture.
Tone: informative, calm, confident. Speaking pace: moderate. Language: Hindi."""

tts.synthesize(text, voice_style, "lecture_output.wav")




---

## 🎧 Output Example

**Input Text:**  
"स्मार्टफोन ने हमारे जीवन के हर पहलू को बदल दिया है..."

**Generated Output:**  
→ Clear, expressive male Hindi narrator voice with balanced tone and mild warmth.

---

## 📊 Performance and Metrics

- **Audio Quality:** 16-bit PCM, normalized to -12 dB RMS  
- **Real-time factor (RTF):** ~1.0–1.5× (on CPU), <0.4× (on GPU)  
- **Sampling Rate:** 22.05 kHz  
- **Output Format:** `WAV`

---


---

## 📦 Dependencies

- `torch`  
- `transformers`  
- `soundfile`  
- `numpy`  
- `google-genai`  
- `parler_tts`  
- `dotenv`

Install all dependencies manually if required:


---

## 🧭 Future Scope

- 🔸 Support for real-time streaming TTS  
- 🔸 Voice cloning via few-shot speaker embeddings  
- 🔸 Improved multilingual noise-robust synthesis  
- 🔸 Web dashboard for text + tone-based generation  

---

Include this file at the root of your project:




