
# Indie-VOICES: Multilingual Audio Generation for Indian Languages 🇮🇳

**Indie-VOICES** is a multilingual text-to-speech (TTS) system designed to generate expressive, human-like voices across 22+ major Indian languages. It combines **Indic-Parler-TTS** for speech synthesis with **Google Gemini** (LLM) for context-aware, expressive voice-tone generation — enabling dynamic, natural-sounding output suitable for podcasts, audiobooks, educational lectures, and more.

---

## 🌐 Overview

This project integrates an LLM-based voice descriptor generator (Gemini) with **Indic-Parler-TTS** (for multilingual audio synthesis). It handles long-form inputs by chunking text, expanding short cues into expressive descriptions, and synthesizing audio in batches to produce seamless final outputs.

### Key Features

- 🔹 Supports 22+ Indian languages (Hindi, Tamil, Bengali, Telugu, Marathi, Kannada, Malayalam, etc.)  
- 🔹 Uses Gemini LLM to expand short voice cues into detailed expressive tone prompts  
- 🔹 Deterministic decoding for stable, reproducible results  
- 🔹 Batch-based long-text generation with silence merging between segments  
- 🔹 RMS normalization and anti-clipping for studio-quality output  
- 🔹 Built-in text preprocessing for sentence-balanced chunking

---

## 🧩 System Architecture

1. Gemini Voice Description Generator  
   - Accepts a short cue (e.g., "friendly Hindi female teacher") and expands it into a detailed tone descriptor (emotion, pitch, pace, prosody).  
   - This descriptor conditions the TTS model for expressive output.

2. Indic-Parler-TTS Synthesis Engine  
   - Tokenizes both voice descriptor and text input.  
   - Processes text in batches using deterministic decoding (no random sampling).  
   - Normalizes and merges generated audio chunks into a single `.wav` file.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
# Clone the repo
git clone https://github.com/IND-Anshuman/Indie-VOICE.git
cd Indie-VOICE

# (Optional) Create and activate a virtual environment
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Recommended Python version: 3.9+  
Supported OS: Linux / Windows / macOS  
GPU: Optional (CUDA-supported GPU recommended for faster synthesis)

---

## 🔐 Environment Setup

Create a `.env` file for API keys and credentials. Use `.env.example` as a template:

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Example `.env` contents:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
# Other keys (if used)
# OTHER_API_KEY=...
```

### 🚀 Authentication

1. Gemini / Google AI credentials  
   - Obtain the Gemini API key from Google AI Studio (or your chosen Gemini provider).  
   - You can export it in your shell or place it in `.env`.

```bash
# macOS / Linux
export GEMINI_API_KEY="your_gemini_api_key"

# Windows PowerShell
$env:GEMINI_API_KEY = "your_gemini_api_key"
```

2. Hugging Face authentication (for model downloads)  

```bash
# Using the Hugging Face CLI
huggingface-cli login
# or
hf auth login
```

When prompted, paste your Hugging Face token from: https://huggingface.co/settings/tokens

---

## 🧠 How It Works (concise)

1. Initialization  
   - The IndicTTSLLM class (or equivalent) initializes a Gemini client (for voice-prompt expansion) and loads the Parler-TTS model/tokenizer (CPU/GPU auto-detection).

2. Voice Cue Interpretation  
   - `generate_voice_description()` sends a short cue to Gemini and receives a refined descriptive prompt (e.g., expressive instructions for pitch, pace, and emotion).

3. Text Processing  
   - The input text is split into sentence-balanced segments for smoother and coherent generation across batches.

4. Audio Generation  
   - For each batch: tokens are prepared (descriptor + text), Parler-TTS generates audio, normalization and anti-clipping are applied, and chunks are merged with a short natural silence (e.g., 150ms).

5. Saving Output  
   - Final audio is exported as WAV (e.g., `output.wav`), ready for playback or further processing.

---

## 🧪 Example Usage

```python
from indie_voices import IndicTTSLLM

# Initialize (example)
tts = IndicTTSLLM(batch_size=2, device="cuda")  # device auto-detects if available

text = """नमस्कार विद्यार्थियों, आज हम स्मार्टफोन के प्रभाव पर चर्चा करेंगे..."""

voice_style = """Generate a clear, engaging male Hindi teacher voice for an educational lecture.
Tone: informative, calm, confident. Speaking pace: moderate. Language: Hindi."""

# Synthesize and save to file
tts.synthesize(text, voice_style, "lecture_output.wav")
```

Notes:
- Adjust batch_size, device, and model-specific parameters as needed.
- For very long texts, increase batch_size or chunk size carefully to balance memory and latency.

---

## 🎧 Output Example

Input:
> "स्मार्टफोन ने हमारे जीवन के हर पहलू को बदल दिया है..."

Generated:
→ Clear, expressive male Hindi narrator voice with balanced tone, mild warmth, and steady pacing.

---

## 📊 Performance & Metrics

- Audio Quality: 16-bit PCM, normalized to -12 dB RMS  
- Real-time factor (RTF): ~1.0–1.5× on CPU, <0.4× on GPU (varies by model & hardware)  
- Sampling Rate: 22.05 kHz (configurable)  
- Output Format: WAV

---

## 📦 Dependencies

Primary libraries used (examples):

- torch
- transformers
- soundfile
- numpy
- google-genai (or relevant Gemini client)
- parler_tts
- python-dotenv

Install manually if required:

```bash
pip install torch transformers soundfile numpy google-genai parler_tts python-dotenv
```

(Exact package names and versions live in requirements.txt — pin versions there for reproducibility.)

---

## 🧭 Future Scope

- Real-time streaming TTS  
- Voice cloning via few-shot speaker embeddings  
- Improved multilingual noise-robust synthesis  
- Web dashboard for interactive text + tone-based generation

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a PR with:

- A clear description of the change
- Reproduction steps for bugs
- Unit tests or example usage for enhancements, where applicable

---

## License

Please refer to the LICENSE file in this repository for license terms.

