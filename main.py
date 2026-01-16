"""
Demo VieNeuSDK v1.1.3 - Full Features Guide
"""

import os
import sys
import datetime
import soundfile as sf
from vieneu import Vieneu
from pathlib import Path

# Thêm thư mục cha vào sys.path để import text_sample
sys.path.insert(0, str(Path(__file__).parent.parent))
from text_sample import TEXT_SAMPLES

def main():
    print("🚀 Initializing VieNeu SDK (v1.1.3)...")
    
    # Initialize SDK
    # Default: "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf" (Speed & CPU Optimized)
    #
    # You can change 'backbone_repo' to balance Quality vs Speed:
    # 1. Better Quality (slower than q4): "pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf"
    # 2. PyTorch 0.3B (Fast, uncompressed): "pnnbao-ump/VieNeu-TTS-0.3B"
    # 3. PyTorch 0.5B (Best Quality, heavy): "pnnbao-ump/VieNeu-TTS"
    # You can also use a GGUF version merged with your own LoRA adapter.
    # See finetuning guide: https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune
    
    # Mode selection:
    # - mode="standard" (Default): Runs locally using GGUF (CPU) or PyTorch
    # - mode="remote": Connects to the LMDeploy server setup above for max speed
    
    tts = Vieneu()
    # Or to use Remote mode (Must start 'lmdeploy serve api_server pnnbao-ump/VieNeu-TTS-0.3B --server-port 23333 --tp 1' in another tab/machine first):
    # tts = Vieneu(model_name="pnnbao-ump/VieNeu-TTS-0.3B", mode="remote", api_base="http://localhost:23333/v1")
    # Example for using Q8 for better quality:
    # tts = Vieneu(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf")

    # ---------------------------------------------------------
    # PART 1: PRESET VOICES
    # ---------------------------------------------------------
    print("\n--- 1. Available Preset Voices ---")
    available_voices = tts.list_preset_voices()
    print("📋 Voices:", available_voices)
    
    # Select a preset voice
    current_voice = tts.get_preset_voice("Binh")
    print("✅ Selected voice: Binh")


    # ---------------------------------------------------------
    # PART 2: CREATE & SAVE CUSTOM VOICE
    # ---------------------------------------------------------
    print("\n--- 2. Create Custom Voice ---")
    
    # Replace with your actual .wav file path and its exact transcript (including punctuation)
    sample_audio = Path(__file__).parent / "example.wav"
    sample_text = "ví dụ 2. tính trung bình của dãy số."

    if sample_audio.exists():
        voice_name = "MyCustomVoice"
        
        print(f"🎙️ Cloning voice from: {sample_audio.name}")
        
        # 'clone_voice' now supports saving directly with 'name' argument
        custom_voice = tts.clone_voice(
            audio_path=sample_audio,
            text=sample_text,
            name=voice_name  # <-- Automatically saves voice to system
        )
        
        print(f"✅ Voice created and saved as: '{voice_name}'")
        
        # Verify functionality
        print("📋 Voice list after adding:", tts.list_preset_voices())
        
        # Switch to new voice
        current_voice = custom_voice
    else:
        print("⚠️ Sample audio not found. Skipping...")


    # ---------------------------------------------------------
    # PART 3: SYNTHESIS WITH ADVANCED PARAMETERS
    # ---------------------------------------------------------
    print("\n--- 3. Speech Synthesis ---")
    
    # Tạo thư mục output với timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / "results" / "VieNeu-TTS" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Total samples: {len(TEXT_SAMPLES)}")
    
    for idx, text in enumerate(TEXT_SAMPLES):
        print(f"\n🎧 Sample {idx + 1}/{len(TEXT_SAMPLES)}: {text[:50]}...")
        
        audio = tts.infer(
            text=text,
            voice=current_voice,
            temperature=1.0,  # Lower (0.1) -> Stable, Higher (1.0+) -> Expressive
            top_k=50
        )
        
        output_file = output_dir / f"sample_{idx + 1}.wav"
        sf.write(str(output_file), audio, 24000)
        print(f"   💾 Saved: {output_file}")

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    tts.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()