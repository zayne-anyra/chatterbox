import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import os
import subprocess
import sys
import warnings
import re

# Suppress the specific LoRACompatibleLinear deprecation warning
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*deprecated.*", category=FutureWarning)

# Suppress torch CUDA sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*", category=FutureWarning)

# Suppress LlamaModel attention implementation warning
warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)

# Suppress past_key_values tuple deprecation warning
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*deprecated.*", category=UserWarning)

# Suppress additional transformers warnings
warnings.filterwarnings("ignore", message=".*LlamaModel.*LlamaSdpaAttention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*We detected that you are passing.*past_key_values.*", category=UserWarning)

# Suppress Gradio audio conversion warning
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*", category=UserWarning)

# More aggressive warning suppression for transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Suppress all warnings containing these key phrases
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*LlamaModel.*")
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")

# Suppress torch/contextlib warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*contextlib.*")

# Suppress torch.load warnings related to TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*weights_only.*argument.*not explicitly passed.*")
warnings.filterwarnings("ignore", message=".*forcing weights_only=False.*")

# Suppress checkpoint manager warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*checkpoint_manager.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*perth.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

def clear_hf_credentials():
    """Clear any cached Hugging Face credentials that might cause 401 errors."""
    try:
        # Clear environment variables
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGINGFACE_HUB_TOKEN', None)
        
        # Try to logout using CLI
        subprocess.run([sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli', 'logout'], 
                      capture_output=True, check=False)
        print("🔧 Cleared Hugging Face credentials")
        return True
    except Exception as e:
        print(f"⚠️ Could not clear HF credentials: {e}")
        return False

def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            error_str = str(e)
            # Check if it's a 401 authentication error
            if "401" in error_str and "Unauthorized" in error_str:
                print("🔧 Detected 401 authentication error. Clearing credentials and retrying...")
                clear_hf_credentials()
                try:
                    # Retry loading the model
                    MODEL = ChatterboxTTS.from_pretrained(DEVICE)
                    if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                        MODEL.to(DEVICE)
                    print(f"Model loaded successfully after clearing credentials. Internal device: {getattr(MODEL, 'device', 'N/A')}")
                except Exception as retry_error:
                    print(f"Error loading model after retry: {retry_error}")
                    raise
            else:
                print(f"Error loading model: {e}")
                raise
    return MODEL

# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_chunk_length: int = 250) -> list[str]:
    """
    Splits text into chunks that respect sentence boundaries and word limits.
    
    Args:
        text: The input text to split
        max_chunk_length: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by commas or phrases
                if len(sentence) > max_chunk_length:
                    # Split by commas or natural breaks
                    parts = re.split(r'[,;]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size_input: int
) -> tuple[int, np.ndarray]:
    """
    Generates TTS audio using the ChatterboxTTS model, handling long text by chunking.

    Args:
        text_input: The text to synthesize (no length limit).
        audio_prompt_path_input: Path to the reference audio file.
        exaggeration_input: Exaggeration parameter for the model.
        temperature_input: Temperature parameter for the model.
        seed_num_input: Random seed (0 for random).
        cfgw_input: CFG/Pace weight.
        chunk_size_input: Maximum characters per chunk.

    Returns:
        A tuple containing the sample rate (int) and the audio waveform (numpy.ndarray).
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    # Split text into manageable chunks
    text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
    
    if len(text_chunks) == 1:
        print(f"Generating audio for text: '{text_input[:50]}...'")
    else:
        print(f"Generating audio in {len(text_chunks)} chunks for text: '{text_input[:50]}...'")
    
    audio_chunks = []
    
    # Temporarily suppress ALL warnings during generation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"Processing chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
            
            wav = current_model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path_input,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                cfg_weight=cfgw_input,
            )
            
            audio_chunks.append(wav.squeeze(0).numpy())
    
    # Concatenate all audio chunks
    if len(audio_chunks) == 1:
        final_audio = audio_chunks[0]
    else:
        # Add small silence between chunks for natural flow
        silence_samples = int(current_model.sr * 0.05)  # 0.05 second silence
        silence = np.zeros(silence_samples)
        
        concatenated_chunks = []
        for i, chunk in enumerate(audio_chunks):
            concatenated_chunks.append(chunk)
            if i < len(audio_chunks) - 1:  # Don't add silence after the last chunk
                concatenated_chunks.append(silence)
        
        final_audio = np.concatenate(concatenated_chunks)
    
    print("Audio generation complete.")
    return (current_model.sr, final_audio)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox TTS Demo
        Generate high-quality speech from text with reference audio styling.
        **Now supports long text with automatic chunking!**
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (any length supported)",
                max_lines=10
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
            )
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=0.5
            )

            with gr.Accordion("More options", open=False):
                chunk_size = gr.Slider(
                    100, 400, step=25, label="Chunk size (characters per chunk)", value=250
                )
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            gr.Markdown(
                """
                **Tips for long text:**
                - Text is automatically split into chunks at sentence boundaries
                - Chunk size controls the maximum characters per chunk
                - Smaller chunks = more consistent quality, larger chunks = fewer seams
                - A small silence is added between chunks for natural flow
                """
            )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            chunk_size,
        ],
        outputs=[audio_output],
    )

demo.launch()
