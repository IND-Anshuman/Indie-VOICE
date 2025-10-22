import os
import re
import torch
import numpy as np
import soundfile as sf
import random
import time
from datetime import datetime
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from google import genai


class IndicTTSLLM:
    """
    Indic Parler-TTS pipeline with Gemini integration.
    Handles long-text chunking, batching, deterministic decoding, and expressive voice synthesis.
    """

    def __init__(
        self,
        tts_model="ai4bharat/indic-parler-tts",
        gemini_model="gemini-2.0-flash",
        batch_size=2,
        device=None,
        gemini_api_key="AIzaSyD9LwoPXrkZx965g1Fn5XQ9TdcNH_cVP-U"
    ):
        start_time = time.time()
        print(f"🔹 [{datetime.now().strftime('%H:%M:%S')}] Starting TTS initialization...")
        
        # CPU-friendly initialization
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🔹 Initializing Indic Parler-TTS on device: {self.device}")

        self.batch_size = batch_size
        self.gemini_model = gemini_model

        # Initialize Gemini client
        gemini_start = time.time()
        gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("❌ Please set your GEMINI_API_KEY environment variable.")
        self.llm = genai.Client(api_key=gemini_api_key)
        print(f"✅ Gemini client initialized in {time.time() - gemini_start:.2f}s")

        # Load model and tokenizers
        model_start = time.time()
        print("🔹 Loading Indic Parler-TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model).to(self.device)
        print(f"✅ Model loaded in {time.time() - model_start:.2f}s")
        
        tokenizer_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(tts_model)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        print(f"✅ Tokenizers loaded in {time.time() - tokenizer_start:.2f}s")
        
        total_time = time.time() - start_time
        print(f"✅ TTS initialization completed in {total_time:.2f}s total")

    # --------------------------
    # Gemini LLM integration
    # --------------------------
    def generate_voice_description(self, short_prompt):
        """
        Expands a short cue like 'energetic female voice' into
        a detailed, expressive tone description.
        """
        llm_start = time.time()
        print(f"🔹 Generating voice description for: '{short_prompt}'...")
        
        system_prompt = (
            """You are a professional voice-design expert specialising in Indian and multilingual voice synthesis.
Below are the profiles of several voice artists and their language expertise:
1. Aarav (Male, Hindi & English) – Calm, confident narrator with a neutral accent, ideal for documentaries.
2. Ananya (Female, Hindi, English & Bengali) – Energetic and expressive, suitable for podcasts and educational content.
3. Ravi (Male, Tamil & Telugu) – Warm and engaging storyteller with smooth emotional transitions.
4. Meera (Female, Malayalam, Kannada & English) – Soft, empathetic tone perfect for assistant voices.
5. Kabir (Male, Marathi & Hindi) – Deep, authoritative, expressive tone used for news or formal narration.
6. Priya (Female, Gujarati, Hindi & English) – Friendly, youthful voice with clear articulation for interactive media.
7. Farhan (Male, Urdu & Hindi) – Polished, emotional tone for poetic or romantic dialogues.
8. Nisha (Female, Punjabi, Hindi & English) – Cheerful and lively, ideal for advertisements or upbeat content.
9. Shashi (Male, Assamese & English) – Light and bright tone, suitable for children’s content and read-aloud.
10. Diya (Female, Odia & Hindi) – Gentle, hospitable voice, ideal for local language announcements.

When generating the expanded description:
- Choose a profile that matches or complements the cue’s language or style.
- Include tone, emotion, pitch, pace, and ambient recording style.
- Keep it concise (2–3 sentences), vivid and realistic.
Generate only the detailed voice description — do not add extra commentary."""
        )

        user_prompt = f"Voice cue: {short_prompt}. Expand it(not descriptively) for a TTS model such that it does not exceeds maximum 256 charecters"

        response = self.llm.models.generate_content(
            model=self.gemini_model,
            contents=[system_prompt, user_prompt]
        )

        description = response.text.strip()
        llm_time = time.time() - llm_start
        print(f"✅ Voice description generated in {llm_time:.2f}s: '{description[:100]}{'...' if len(description) > 100 else ''}'")
        return description

    def _ensure_attention_mask(self, inputs):
        """Ensure attention_mask exists and is properly shaped."""
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs

    def _validate_sequence_length(self, inputs, max_length=4096):
        """Validate and truncate sequence if too long."""
        seq_len = inputs["input_ids"].shape[-1]
        if seq_len > max_length:
            print(f"⚠️ Truncating sequence from {seq_len} to {max_length} tokens")
            inputs["input_ids"] = inputs["input_ids"][:, :max_length]
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        return inputs

    # --------------------------
    # Text splitting
    # --------------------------
    def _split_text(self, text, max_len=250):
        """Split long text into balanced sentence chunks."""
        chunks = re.split(r'(?<=[।.!?\n])\s', text)
        merged, buff = [], ""
        for chunk in chunks:
            if len(buff) + len(chunk) < max_len:
                buff += chunk + " "
            else:
                merged.append(buff.strip())
                buff = chunk + " "
        if buff:
            merged.append(buff.strip())
        return merged

    # --------------------------
    # Audio normalization helper
    # --------------------------
    def _normalize_rms(self, wav, target_db=-12.0, prevent_clipping=True):
        """Normalize audio with clipping prevention."""
        # Calculate RMS and target level
        rms = np.sqrt(np.mean(wav ** 2))
        target = 10 ** (target_db / 20.0)
        
        if rms > 0:
            # Calculate gain needed
            gain = target / (rms + 1e-9)
            wav = wav * gain
            
            # Prevent clipping by reducing gain if needed
            if prevent_clipping:
                peak = np.max(np.abs(wav))
                if peak > 0.95:  # Leave headroom
                    clip_gain = 0.95 / peak
                    wav = wav * clip_gain
                    print(f"  🔊 Applied anti-clipping gain: {clip_gain:.3f}")
        
        return wav

    # --------------------------
    # Speech synthesis pipeline
    # --------------------------
    def synthesize(self, text, voice_style, output_path="output.wav"):
        """
        Synthesizes TTS output for given text and short voice cue.
        Implements deterministic decoding and stable conditioning.
        """
        synthesis_start = time.time()
        print(f"\n🎤 Starting TTS synthesis...")
        print(f"📝 Text length: {len(text)} characters")
        print(f"🎭 Voice style: '{voice_style}'")
        print(f"📁 Output path: {output_path}")
        
        print(f"\n🎤 Expanding short cue '{voice_style}' using Gemini...")
        full_description = self.generate_voice_description(voice_style)
        print(f"🗣️ Generated detailed voice description:\n{full_description}\n")

        # Text chunking
        chunk_start = time.time()
        text_chunks = self._split_text(text)
        print(f"🧩 Split text into {len(text_chunks)} chunks in {time.time() - chunk_start:.3f}s\n")

        # Tokenize description with proper attention mask
        tokenize_start = time.time()
        print("🔤 Tokenizing voice description...")
        desc_inputs = self.desc_tokenizer(
            full_description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Ensure attention mask exists and validate sequence length
        desc_inputs = self._ensure_attention_mask(desc_inputs)
        desc_inputs = self._validate_sequence_length(desc_inputs, max_length=256)
        print(f"✅ Description tokenized in {time.time() - tokenize_start:.3f}s (length: {desc_inputs['input_ids'].shape[-1]})")
        
        # Fix random seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("\n🔄 Starting batch synthesis...")
        synthesis_loop_start = time.time()
        audios = []
        total_batches = (len(text_chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, start in enumerate(range(0, len(text_chunks), self.batch_size)):
            batch_start = time.time()
            batch = text_chunks[start:start + self.batch_size]
            print(f"\n📦 Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks)...")
            
            # Tokenize description for this batch
            desc_token_start = time.time()
            desc_inputs = self.desc_tokenizer(
                full_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            desc_inputs = self._ensure_attention_mask(desc_inputs)
            desc_inputs = self._validate_sequence_length(desc_inputs, 256)
            print(f"  🔤 Description tokenized in {time.time() - desc_token_start:.3f}s")
            
            # Tokenize batch text
            text_token_start = time.time()
            batch_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            batch_inputs = self._ensure_attention_mask(batch_inputs)
            batch_inputs = self._validate_sequence_length(batch_inputs, 256)
            print(f"  📝 Text tokenized in {time.time() - text_token_start:.3f}s")

            # Calculate safe max_new_tokens to prevent position overflow
            max_pos = getattr(self.model.config, 'max_position_embeddings', 4096)
            current_len = max(desc_inputs.input_ids.shape[-1], batch_inputs.input_ids.shape[-1])
            safe_max_tokens = min(8000, max_pos - current_len - 50)  # Leave buffer
            print(f"  📏 Max position: {max_pos}, Current length: {current_len}, Safe tokens: {safe_max_tokens}")
            
            if safe_max_tokens <= 0:
                print(f"⚠️ Input too long, skipping batch")
                continue

            # Generate audio with timing
            generation_start = time.time()
            print(f"  🎵 Generating audio...")
            with torch.no_grad():
                batch_audio = self.model.generate(
                    input_ids=desc_inputs.input_ids.repeat(len(batch), 1),
                    attention_mask=desc_inputs.attention_mask.repeat(len(batch), 1),
                    prompt_input_ids=batch_inputs.input_ids,
                    prompt_attention_mask=batch_inputs.attention_mask,
                    do_sample=False,          # deterministic greedy decoding
                    use_cache=True,
                    max_new_tokens=safe_max_tokens,
                )
            generation_time = time.time() - generation_start
            print(f"  ✅ Audio generated in {generation_time:.2f}s")

            # Post-process audio
            postprocess_start = time.time()
            for wav in batch_audio.cpu().numpy():
                wav = wav.squeeze()
                
                # Check for problematic values
                if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
                    print(f"  ⚠️ Found NaN/Inf values, cleaning...")
                    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Remove any DC offset
                wav = wav - np.mean(wav)
                
                # Apply gentle normalization
                wav = self._normalize_rms(wav, target_db=-12.0, prevent_clipping=True)
                audios.append(wav)
            postprocess_time = time.time() - postprocess_start
            
            batch_time = time.time() - batch_start
            print(f"  📊 Batch {batch_idx + 1} completed in {batch_time:.2f}s (generation: {generation_time:.2f}s, postprocess: {postprocess_time:.3f}s)")

        synthesis_loop_time = time.time() - synthesis_loop_start
        print(f"\n✅ All batches completed in {synthesis_loop_time:.2f}s")

        # Merge audio clips with small silence
        merge_start = time.time()
        print("🔗 Merging audio clips...")
        sr = self.model.config.sampling_rate
        silence = np.zeros(int(0.15 * sr))
        
        if len(audios) > 0:
            combined = audios[0]
            for clip in audios[1:]:
                combined = np.concatenate([combined, silence, clip])
                
            # Final normalization to ensure consistent output level
            combined = self._normalize_rms(combined, target_db=-12.0, prevent_clipping=True)
            print(f"✅ Audio merged in {time.time() - merge_start:.3f}s")
        else:
            print("❌ No audio clips to merge!")
            return None

        # Ensure WAV output and save
        save_start = time.time()
        if not output_path.endswith(".wav"):
            output_path += ".wav"

        sf.write(output_path, combined, sr)
        save_time = time.time() - save_start
        
        total_time = time.time() - synthesis_start
        audio_duration = len(combined) / sr
        realtime_factor = audio_duration / total_time if total_time > 0 else 0
        
        # Final audio analysis
        final_peak = np.max(np.abs(combined))
        final_rms = np.sqrt(np.mean(combined ** 2))
        final_peak_db = 20 * np.log10(final_peak + 1e-10)
        final_rms_db = 20 * np.log10(final_rms + 1e-10)
        
        print(f"💾 Audio saved in {save_time:.3f}s")
        print(f"🎉 TTS synthesis completed!")
        print(f"📊 Total time: {total_time:.2f}s | Audio duration: {audio_duration:.1f}s | RTF: {realtime_factor:.2f}x")
        print(f"🔊 Final audio levels: Peak={final_peak:.3f} ({final_peak_db:.1f}dB), RMS={final_rms:.3f} ({final_rms_db:.1f}dB)")
        print(f"✅ TTS audio saved at: {output_path}")
        return output_path


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = "gemini_api_key"

    tts_pipe = IndicTTSLLM(batch_size=2)

    text_input = (
        '''नमस्कार विद्यार्थियों,
आज हम एक अत्यंत रोचक विषय पर चर्चा करने जा रहे हैं — स्मार्टफोन का हमारे जीवन पर प्रभाव﻿। पिछले एक दशक में, स्मार्टफोन ने हमारे जीवन के हर पहलू को बदल दिया है। संचार, शिक्षा, मनोरंजन, और व्यवसाय — हर क्षेत्र में इसकी भूमिका बढ़ती जा रही है।
सबसे पहले बात करते हैं शिक्षा की। अब विद्यार्थी ऑनलाइन व्याख्यान सुन सकते हैं, नोट्स डाउनलोड कर सकते हैं, और दुनिया के किसी भी कोने से अध्ययन कर सकते हैं। यह तकनीकी क्रांति शिक्षा को अधिक सुलभ बना रही है।
लेकिन इसके साथ ही हमें इसके नकारात्मक पक्ष को भी समझना चाहिए। स्मार्टफोन का अत्यधिक उपयोग हमारी एकाग्रता को कम करता है और आँखों पर बुरा प्रभाव डालता है। इसलिए तकनीक का संतुलित उपयोग ही हमारे लिए उचित है।
अंत में, यही कहा जा सकता है कि स्मार्टफोन एक शक्तिशाली उपकरण है — यह हमें आगे भी ले जा सकता है या पीछे भी खींच सकता है, यह इस बात पर निर्भर करता है कि हम इसका उपयोग किस प्रकार करते हैं।'''
    )

    voice_cue = '''Generate a clear, engaging male Hindi teacher voice for an educational lecture.  
Tone: informative, calm, and confident.  
Speaking pace: moderate, with natural pauses after key ideas.  
Emotion: thoughtful and explanatory.  
Language: Hindi.'''

    tts_pipe.synthesize(text_input, voice_cue, "gemini_tts_output.wav")
