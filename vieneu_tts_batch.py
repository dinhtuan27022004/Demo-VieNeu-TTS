"""
VieNeu TTS Batch Synthesis
Combines all notebook functionality into a single function for batch text-to-speech synthesis.
"""

import sys
import datetime
import soundfile as sf
from pathlib import Path
from vieneu import Vieneu


def run_vieneu_tts_batch(
    text_samples=None,
    output_base_dir="/home/lamquy/Project/TTS/results/VieNeu-TTS",
    notebook_dir="/home/lamquy/Project/TTS/VieNeu-TTS",
    preset_voice="Binh",
    use_custom_voice=False,
    sample_audio_path=None,
    sample_audio_text=None,
    custom_voice_name="MyCustomVoice",
    temperature=1.0,
    top_k=50,
    sample_rate=24000
):
    """
    Run batch TTS synthesis using VieNeu SDK.
    
    Args:
        text_samples (list): List of text strings to synthesize. If None, will import from text_sample.py
        output_base_dir (str|Path): Base directory for output files
        notebook_dir (str|Path): Directory containing the notebook and examples
        preset_voice (str): Name of preset voice to use (e.g., "Binh")
        use_custom_voice (bool): Whether to clone and use a custom voice
        sample_audio_path (str|Path): Path to sample audio for voice cloning
        sample_audio_text (str): Transcript of the sample audio
        custom_voice_name (str): Name to save the cloned voice as
        temperature (float): Temperature for synthesis (0.1=stable, 1.0+=expressive)
        top_k (int): Top-k sampling parameter
        sample_rate (int): Audio sample rate for output files
        
    Returns:
        Path: Directory containing the generated audio files
    """
    
    # ====== SETUP & CONFIGURATION ======
    notebook_dir = Path(notebook_dir)
    output_base_dir = Path(output_base_dir)
    
    # Import text samples if not provided
    if text_samples is None:
        sys.path.insert(0, '/home/lamquy/Project/TTS')
        from text_sample import TEXT_SAMPLES
        text_samples = TEXT_SAMPLES
    
    # Set sample audio path defaults
    if sample_audio_path is None:
        sample_audio_path = notebook_dir / "examples" / "audio_ref" / "example.wav"
    else:
        sample_audio_path = Path(sample_audio_path)
    
    if sample_audio_text is None:
        sample_audio_text = "ví dụ 2. tính trung bình của dãy số."
    
    # ====== INITIALIZE VieNeu SDK ======
    print("🚀 Initializing VieNeu SDK...")
    tts = Vieneu()
    print("✅ SDK initialized successfully")
    
    try:
        # ====== SELECT VOICE ======
        # List all available preset voices
        available_voices = tts.list_preset_voices()
        print("📋 Available preset voices:", available_voices)
        
        # Select a preset voice
        current_voice = tts.get_preset_voice(preset_voice)
        print(f"✅ Selected voice: {preset_voice}")
        
        # ====== CLONE CUSTOM VOICE (OPTIONAL) ======
        if use_custom_voice:
            # Check if sample audio exists
            if sample_audio_path.exists():
                print(f"🎙️ Cloning voice from: {sample_audio_path.name}")
                
                # Clone voice and save with custom name
                custom_voice = tts.clone_voice(
                    audio_path=sample_audio_path,
                    text=sample_audio_text,
                    name=custom_voice_name
                )
                
                print(f"✅ Voice cloned and saved as: '{custom_voice_name}'")
                
                # Switch to the new custom voice
                current_voice = custom_voice
                
                # Verify it was added to the voice list
                print("📋 Updated voice list:", tts.list_preset_voices())
            else:
                print(f"⚠️ Sample audio not found at: {sample_audio_path}")
                print("   Continuing with preset voice...")
        
        # ====== BATCH SPEECH SYNTHESIS ======
        # Create output directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        print(f"📝 Total samples to process: {len(text_samples)}")
        
        # Process each text sample
        for idx, text in enumerate(text_samples, start=1):
            # Show progress
            print(f"\n🎧 Sample {idx}/{len(text_samples)}: {text[:50]}...")
            
            # Generate audio
            audio = tts.infer(
                text=text,
                voice=current_voice,
                temperature=temperature,
                top_k=top_k
            )
            
            # Save to file
            output_file = output_dir / f"sample_{idx}.wav"
            sf.write(str(output_file), audio, sample_rate)
            print(f"   💾 Saved: {output_file}")
        
        print("\n✅ All samples processed successfully!")
        
        return output_dir
        
    finally:
        # ====== CLEANUP ======
        # Close the TTS engine
        tts.close()
        print("✅ TTS engine closed")


if __name__ == "__main__":
    # Example usage with default settings (preset voice)
    output_dir = run_vieneu_tts_batch()
    print(f"\n📂 All audio files saved to: {output_dir}")
    
    # Example usage with custom voice cloning
    # output_dir = run_vieneu_tts_batch(
    #     use_custom_voice=True,
    #     custom_voice_name="MyVoice"
    # )
