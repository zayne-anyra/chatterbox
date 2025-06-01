import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import os
import subprocess
import sys
import warnings

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

def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float
) -> tuple[int, np.ndarray]:
    """
    Generates TTS audio using the ChatterboxTTS model.

    Args:
        text_input: The text to synthesize (max 300 characters).
        audio_prompt_path_input: Path to the reference audio file.
        exaggeration_input: Exaggeration parameter for the model.
        temperature_input: Temperature parameter for the model.
        seed_num_input: Random seed (0 for random).
        cfgw_input: CFG/Pace weight.

    Returns:
        A tuple containing the sample rate (int) and the audio waveform (numpy.ndarray).
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    
    # Temporarily suppress ALL warnings during generation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav = current_model.generate(
            text_input[:300],  # Truncate text to max chars
            audio_prompt_path=audio_prompt_path_input,
            exaggeration=exaggeration_input,
            temperature=temperature_input,
            cfg_weight=cfgw_input,
        )
    
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox TTS Demo
        Generate high-quality speech from text with reference audio styling.
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
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
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

demo.launch()