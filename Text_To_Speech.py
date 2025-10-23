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
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the backend directory
import pathlib
backend_root = pathlib.Path(__file__).parent.parent.parent
env_path = backend_root / '.env'
load_dotenv(env_path)


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
        gemini_api_key="GEMINI_API_KEY"
    ):
        start_time = time.time()
        print(f"🔹 [{datetime.now().strftime('%H:%M:%S')}] Starting TTS initialization...")
        
        # Smart device selection with memory optimization
        if device is None:
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🔹 Detected GPU with {gpu_memory:.1f}GB memory")
                
                # Your integrated GPU setup - use GPU with optimizations
                if gpu_memory >= 8.0:  # Your 8.8GB setup
                    self.device = "cuda:0"
                    self.use_mixed_precision = True
                    print("🚀 Using GPU with mixed precision optimization")
                elif gpu_memory >= 4.0:
                    self.device = "cuda:0" 
                    self.use_mixed_precision = True
                    print("⚡ Using GPU with memory optimization")
                else:
                    self.device = "cpu"
                    self.use_mixed_precision = False
                    print("💻 GPU memory insufficient, using CPU")
            else:
                self.device = "cpu"
                self.use_mixed_precision = False
                print("💻 No GPU detected, using CPU")
        else:
            self.device = device
            self.use_mixed_precision = device.startswith("cuda")
        
        print(f"🔹 Initializing Indic Parler-TTS on device: {self.device}")

        self.batch_size = batch_size
        self.gemini_model = gemini_model
        
        # Auto-adjust batch size based on device capabilities
        if self.device.startswith("cuda"):
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 8.0:  # Your setup
                self.batch_size = min(batch_size, 3)  # Conservative for integrated GPU
                print(f"🔧 GPU batch size optimized to: {self.batch_size}")
            elif gpu_memory >= 4.0:
                self.batch_size = min(batch_size, 2)
                print(f"🔧 GPU batch size optimized to: {self.batch_size}")
            else:
                self.batch_size = 1
                print(f"🔧 GPU batch size limited to: {self.batch_size}")
        else:
            # CPU can handle larger batches due to system RAM
            self.batch_size = batch_size
            print(f"💻 CPU batch size: {self.batch_size}")

        # Initialize Gemini client
        gemini_start = time.time()
        gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("❌ Please set your GEMINI_API_KEY environment variable.")
        self.llm = genai.Client(api_key=gemini_api_key)
        print(f"✅ Gemini client initialized in {time.time() - gemini_start:.2f}s")

        # Load model and tokenizers with GPU optimization
        model_start = time.time()
        print("🔹 Loading Indic Parler-TTS model...")
        
        if self.device.startswith("cuda"):
            # GPU optimization for your integrated GPU setup
            print("🔧 Applying GPU memory optimizations...")
            torch.cuda.empty_cache()  # Clear any existing GPU memory
            
            # Load model with optimizations
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                tts_model,
                torch_dtype=torch.float16 if self.use_mixed_precision else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Monitor GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        else:
            # CPU loading
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

        user_prompt = f"Voice cue: {short_prompt}. Make a small prompt for a TTS model such that it does not exceeds maximum 256 charecters"

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
    def _normalize_audio(self, wav, target_peak=0.95):
        """
        Simple peak normalization - just scales to target volume.
        No clipping, no filtering, no processing - pure volume adjustment.
        """
        peak = np.abs(wav).max()
        if peak > 0:
            return wav * (target_peak / peak)
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
        desc_inputs_initial = self.desc_tokenizer(
            full_description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # CRITICAL: Explicitly ensure attention mask exists
        if "attention_mask" not in desc_inputs_initial:
            desc_inputs_initial["attention_mask"] = torch.ones_like(desc_inputs_initial["input_ids"])
        
        desc_inputs_initial = self._validate_sequence_length(desc_inputs_initial, max_length=256)
        print(f"✅ Description tokenized in {time.time() - tokenize_start:.3f}s (length: {desc_inputs_initial['input_ids'].shape[-1]})")
        print(f"   Attention mask present: {'attention_mask' in desc_inputs_initial}, shape: {desc_inputs_initial['attention_mask'].shape if 'attention_mask' in desc_inputs_initial else 'N/A'}")
        
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
            
            # CRITICAL: Explicitly ensure attention mask exists
            if "attention_mask" not in desc_inputs:
                desc_inputs["attention_mask"] = torch.ones_like(desc_inputs["input_ids"])
            
            desc_inputs = self._validate_sequence_length(desc_inputs, 256)
            print(f"  🔤 Description tokenized in {time.time() - desc_token_start:.3f}s")
            print(f"      Description attention_mask shape: {desc_inputs['attention_mask'].shape}")
            
            # Tokenize batch text
            text_token_start = time.time()
            batch_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # CRITICAL: Explicitly ensure attention mask exists
            if "attention_mask" not in batch_inputs:
                batch_inputs["attention_mask"] = torch.ones_like(batch_inputs["input_ids"])
                
            batch_inputs = self._validate_sequence_length(batch_inputs, 256)
            print(f"  📝 Text tokenized in {time.time() - text_token_start:.3f}s")
            print(f"      Text attention_mask shape: {batch_inputs['attention_mask'].shape}")

            # Calculate safe max_new_tokens to prevent position overflow
            max_pos = getattr(self.model.config, 'max_position_embeddings', 4096)
            current_len = max(desc_inputs["input_ids"].shape[-1], batch_inputs["input_ids"].shape[-1])
            safe_max_tokens = min(8000, max_pos - current_len - 50)  # Leave buffer
            print(f"  📏 Max position: {max_pos}, Current length: {current_len}, Safe tokens: {safe_max_tokens}")
            
            if safe_max_tokens <= 0:
                print(f"⚠️ Input too long, skipping batch")
                continue

            # Generate audio with timing and GPU optimization
            generation_start = time.time()
            print(f"  🎵 Generating audio...")
            
            # Prepare inputs with proper attention masks
            desc_input_ids = desc_inputs["input_ids"].repeat(len(batch), 1)
            desc_attention_mask = desc_inputs["attention_mask"].repeat(len(batch), 1)
            
            # Debug: Verify attention masks before generation
            print(f"      desc_input_ids shape: {desc_input_ids.shape}")
            print(f"      desc_attention_mask shape: {desc_attention_mask.shape}")
            print(f"      prompt_input_ids shape: {batch_inputs['input_ids'].shape}")
            print(f"      prompt_attention_mask shape: {batch_inputs['attention_mask'].shape}")
            
            # Memory monitoring for GPU
            if self.device.startswith("cuda"):
                pre_gen_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"    Pre-generation GPU memory: {pre_gen_memory:.2f}GB")
            
            with torch.no_grad():
                # Use autocast for mixed precision on GPU
                if self.use_mixed_precision and self.device.startswith("cuda"):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        batch_audio = self.model.generate(
                            input_ids=desc_input_ids,
                            attention_mask=desc_attention_mask,
                            prompt_input_ids=batch_inputs["input_ids"],
                            prompt_attention_mask=batch_inputs["attention_mask"],
                            do_sample=True,           # CHANGED: Use sampling for better quality
                            temperature=1.0,          # Control randomness
                            use_cache=True,
                            max_new_tokens=safe_max_tokens,
                        )
                else:
                    batch_audio = self.model.generate(
                        input_ids=desc_input_ids,
                        attention_mask=desc_attention_mask,
                        prompt_input_ids=batch_inputs["input_ids"],
                        prompt_attention_mask=batch_inputs["attention_mask"],
                        do_sample=True,           # CHANGED: Use sampling for better quality
                        temperature=1.0,          # Control randomness
                        use_cache=True,
                        max_new_tokens=safe_max_tokens,
                    )
            
            generation_time = time.time() - generation_start
            
            # Post-generation memory check
            if self.device.startswith("cuda"):
                post_gen_memory = torch.cuda.memory_allocated(0) / (1024**3)
                peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
                print(f"    Post-generation GPU memory: {post_gen_memory:.2f}GB (Peak: {peak_memory:.2f}GB)")
                torch.cuda.empty_cache()  # Clean up
            print(f"  ✅ Audio generated in {generation_time:.2f}s")

            # Post-process audio - model.generate() already returns decoded waveforms
            postprocess_start = time.time()
            
            print(f"      Batch audio type: {type(batch_audio)}, shape: {batch_audio.shape}")
            
            # The generate() method already returns decoded audio waveforms, not codes
            # No need for additional decoding
            for idx, wav in enumerate(batch_audio.cpu().numpy()):
                wav = wav.squeeze()
                
                # Debug: Check raw audio before normalization
                print(f"      Raw audio {idx}: shape={wav.shape}, dtype={wav.dtype}")
                print(f"      Raw audio stats: min={wav.min():.6f}, max={wav.max():.6f}, mean={wav.mean():.6f}, std={wav.std():.6f}")
                
                # Check if audio is valid (not just zeros or constant)
                if wav.std() < 0.001:  # Very low variation indicates a problem
                    print(f"      WARNING: Audio {idx} has very low variation (std={wav.std():.6f})")
                    print(f"      This indicates the model may not be generating proper speech.")
                    print(f"      Possible causes:")
                    print(f"        - Text/description mismatch with model training")
                    print(f"        - Model not properly loaded")
                    print(f"        - Incorrect generation parameters")
                
                # Don't normalize if audio is essentially flat
                if wav.std() > 0.001:
                    wav = self._normalize_audio(wav, target_peak=0.9)
                    print(f"      Normalized audio {idx}: min={wav.min():.6f}, max={wav.max():.6f}, mean={wav.mean():.6f}, std={wav.std():.6f}")
                else:
                    print(f"      WARNING: Skipping normalization - audio is nearly flat/silent")
                
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
            
            # Final simple normalization for audibility
            combined = self._normalize_audio(combined, target_peak=0.9)
            
            print(f"✅ Audio merged in {time.time() - merge_start:.3f}s")
            
            # Audio level check
            peak = np.abs(combined).max()
            rms = np.sqrt(np.mean(combined ** 2))
            print(f"📊 Final audio - Peak: {peak:.3f}, RMS: {rms:.4f}")
        else:
            print("❌ No audio clips to merge!")
            return None
        # Ensure WAV output and save - NO PROCESSING
        save_start = time.time()
        if not output_path.endswith(".wav"):
            output_path += ".wav"

        sf.write(output_path, combined, sr)
        save_time = time.time() - save_start
        
        total_time = time.time() - synthesis_start
        audio_duration = len(combined) / sr
        realtime_factor = audio_duration / total_time if total_time > 0 else 0
        
        print(f"💾 Audio saved in {save_time:.3f}s")
        print(f"🎉 TTS synthesis completed!")
        print(f"📊 Total time: {total_time:.2f}s | Audio duration: {audio_duration:.1f}s | RTF: {realtime_factor:.2f}x")
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
अंत में, यही कहा जा सकता है कि स्मार्टफोन एक शक्तिशाली उपकरण है — यह हमें आगे भी ले जा सकता है या पीछे भी खींच सकता है, यह इस बात पर निर्भर करता है कि हम इसका उपयोग किस प्रकार करते हैं।'''
    )

    voice_cue = '''Generate a clear, engaging male Hindi teacher voice for an educational lecture.  
Tone: informative, calm, and confident.  
Speaking pace: moderate, with natural pauses after key ideas.  
Emotion: thoughtful and explanatory.  
Language: Hindi.'''

    tts_pipe.synthesize(text_input, voice_cue, "gemini_tts_output.wav")

