"""
TTS + WER Evaluation Script
Combines VieNeu TTS synthesis with automatic WER evaluation using Whisper ASR.
"""

import sys
import datetime
import soundfile as sf
from pathlib import Path
from vieneu import Vieneu
import whisper
import matplotlib.pyplot as plt
from jiwer import wer, process_words


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
    """Run batch TTS synthesis using VieNeu SDK."""
    
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
        available_voices = tts.list_preset_voices()
        print("📋 Available preset voices:", available_voices)
        
        current_voice = tts.get_preset_voice(preset_voice)
        print(f"✅ Selected voice: {preset_voice}")
        
        # ====== CLONE CUSTOM VOICE (OPTIONAL) ======
        if use_custom_voice:
            if sample_audio_path.exists():
                print(f"🎙️ Cloning voice from: {sample_audio_path.name}")
                
                custom_voice = tts.clone_voice(
                    audio_path=sample_audio_path,
                    text=sample_audio_text,
                    name=custom_voice_name
                )
                
                print(f"✅ Voice cloned and saved as: '{custom_voice_name}'")
                current_voice = custom_voice
                print("📋 Updated voice list:", tts.list_preset_voices())
            else:
                print(f"⚠️ Sample audio not found at: {sample_audio_path}")
                print("   Continuing with preset voice...")
        
        # ====== BATCH SPEECH SYNTHESIS ======
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        print(f"📝 Total samples to process: {len(text_samples)}")
        
        # Process each text sample
        for idx, text in enumerate(text_samples, start=1):
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
        tts.close()
        print("✅ TTS engine closed")


def evaluate_wer(text_samples, output_dir, asr_model_name="large"):
    """
    Evaluate WER for generated audio files using Whisper ASR.
    
    Args:
        text_samples: List of reference text strings
        output_dir: Directory containing generated audio files
        asr_model_name: Whisper model to use for ASR
        
    Returns:
        List of dictionaries containing WER results
    """
    print("\n" + "="*60)
    print("STEP 2: WER EVALUATION")
    print("="*60)
    
    # Load Whisper model once
    print(f"\n🔄 Loading Whisper ASR model ({asr_model_name})...")
    whisper_model = whisper.load_model(asr_model_name)
    print("✅ Whisper model loaded")
    
    # Store results
    wer_results = []
    
    # Evaluate each generated audio file
    for idx, ref_text in enumerate(text_samples, start=1):
        audio_file = output_dir / f"sample_{idx}.wav"
        
        print(f"\n{'='*60}")
        print(f"Sample {idx}/{len(text_samples)}")
        print(f"{'='*60}")
        print(f"📝 Reference: {ref_text[:80]}..." if len(ref_text) > 80 else f"📝 Reference: {ref_text}")
        print(f"🎧 Audio: {audio_file.name}")
        
        # Transcribe with Whisper
        print("🔄 Transcribing...")
        result = whisper_model.transcribe(str(audio_file), language="vi")
        hyp_text = result["text"].strip()
        
        # Normalize
        ref_normalized = ref_text.lower().strip()
        hyp_normalized = hyp_text.lower().strip()
        
        # Compute WER
        wer_score = wer(ref_normalized, hyp_normalized)
        
        # Error breakdown
        details = process_words(ref_normalized, hyp_normalized)
        error_counts = {
            "Correct": details.hits,
            "Substitution": details.substitutions,
            "Deletion": details.deletions,
            "Insertion": details.insertions,
        }
        
        # Print results
        print(f"\n✅ Transcription complete!")
        print(f"🎯 Hypothesis: {hyp_text[:80]}..." if len(hyp_text) > 80 else f"🎯 Hypothesis: {hyp_text}")
        print(f"\n📊 WER Score: {wer_score:.4f} ({wer_score*100:.2f}%)")
        print(f"📈 Error Breakdown:")
        for error_type, count in error_counts.items():
            print(f"   - {error_type}: {count}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(error_counts.keys(), error_counts.values(), color=['green', 'orange', 'red', 'purple'])
        ax.set_title(f"Sample {idx} - WER: {wer_score*100:.2f}%", fontsize=14, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel("Error Type", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Store result
        wer_results.append({
            "sample_id": idx,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "wer": wer_score,
            "error_breakdown": error_counts
        })
    
    return wer_results


def plot_wer_summary(wer_results):
    """Plot summary of WER results."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_wer = sum(r["wer"] for r in wer_results) / len(wer_results)
    print(f"\n📊 Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"\n📋 Individual WER scores:")
    for r in wer_results:
        print(f"   Sample {r['sample_id']}: {r['wer']*100:.2f}%")
    
    # Final summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_ids = [r["sample_id"] for r in wer_results]
    wer_scores = [r["wer"] * 100 for r in wer_results]
    bars = ax.bar(sample_ids, wer_scores, color='steelblue')
    ax.axhline(y=avg_wer*100, color='red', linestyle='--', label=f'Average: {avg_wer*100:.2f}%')
    ax.set_title("WER Scores Across All Samples", fontsize=16, fontweight='bold')
    ax.set_xlabel("Sample ID", fontsize=12)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_xticks(sample_ids)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    # Text samples to synthesize and evaluate
    TEXT_SAMPLES = [
        # Bảng chữ cái tiếng Việt
        "A Ă Â B C D Đ E Ê G H I K L M N O Ô Ơ P Q R S T U Ư V X Y",
        
        # Đoạn văn dài - test độ ổn định
        "Tiếng Việt là ngôn ngữ giàu thanh điệu và hình ảnh, phản ánh đời sống tinh tế của con người Việt Nam.",
        
        # Tên riêng và địa danh
        "Nguyễn Ái Quốc đã từng viết về những chuyến chu du dài dằng dặc qua châu Âu giữa mùa đông rét mướt.",
        
        # Phụ âm khó (ch, tr, s, x)
        "Chị Trúc nhặt nhạnh từng chiếc chén sứ sứt sẹo trên chiếc chõng tre trước hiên nhà.",
        
        # Câu ngắn
        "Đây là chữ g",
        "Nếu bạn không biết mình đang ở đâu, thì bất cứ con đường nào cũng sẽ dẫn bạn đến đó.",
    ]
    
    # ========== STEP 1: Run TTS Batch Synthesis ==========
    print("="*60)
    print("STEP 1: TTS SYNTHESIS")
    print("="*60)
    
    output_dir = run_vieneu_tts_batch(
        text_samples=TEXT_SAMPLES,
        output_base_dir="/home/lamquy/Project/TTS/results/VieNeu-TTS",
        notebook_dir="/home/lamquy/Project/TTS/VieNeu-TTS",
        preset_voice="Binh",
        use_custom_voice=False,
        temperature=1.0,
        top_k=50,
        sample_rate=24000
    )
    
    print(f"\n📂 All audio files saved to: {output_dir}")
    
    # ========== STEP 2: WER Evaluation ==========
    wer_results = evaluate_wer(TEXT_SAMPLES, output_dir, asr_model_name="large")
    
    # ========== STEP 3: Summary ==========
    plot_wer_summary(wer_results)
