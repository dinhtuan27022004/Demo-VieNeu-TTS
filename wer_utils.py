import json
import datetime
from pathlib import Path

import whisper
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F

from jiwer import wer, cer, process_words, process_characters
from speechbrain.inference.speaker import SpeakerRecognition

TARGET_SR = 16000

def _load_wav_mono_resample(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()  # [1, T]

class SpeakerSimECAPA:
    """
    Lazy-load ECAPA model 1 lần, dùng lại nhiều lần (đỡ tải lại mỗi file).
    """
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    @torch.inference_mode()
    def cosine_similarity(self, ref_audio_path: str, gen_audio_path: str) -> float:
        ref = _load_wav_mono_resample(ref_audio_path).squeeze(0).unsqueeze(0).to(self.device)  # [1,T]
        gen = _load_wav_mono_resample(gen_audio_path).squeeze(0).unsqueeze(0).to(self.device)

        emb_ref = self.model.encode_batch(ref).squeeze()
        emb_gen = self.model.encode_batch(gen).squeeze()

        sim = F.cosine_similarity(emb_ref, emb_gen, dim=0).item()
        return float(sim)

# Khởi tạo global instance sau khi class được định nghĩa
spk_sim = None  # Lazy init để tránh load model khi import

def get_speaker_sim_model():
    global spk_sim
    if spk_sim is None:
        spk_sim = SpeakerSimECAPA()
    return spk_sim

def metric_calculate(
    audio_path,
    ref_text,
    model_name,
    asr_model_name="large",
    language="vi",
    result_dir="../results",
    ref_audio_path=None,         # ✅ NEW: audio giọng gốc để so speaker similarity
    speaker_sim_model=None,      # ✅ NEW: truyền instance SpeakerSimECAPA để reuse
):
    """
    Tính toán WER + CER (+ Speaker Similarity nếu có ref_audio_path) và lưu JSON
    """
    # Load ASR model
    model = whisper.load_model(asr_model_name)

    # Transcribe audio
    result = model.transcribe(str(audio_path), language=language)
    hyp_text = result["text"].strip()

    # Normalize texts
    ref_text_normalized = ref_text.lower().strip()
    hyp_text_normalized = hyp_text.lower().strip()

    # Compute WER + CER
    wer_score = wer(ref_text_normalized, hyp_text_normalized)
    cer_score = cer(ref_text_normalized, hyp_text_normalized)

    # Word-level error breakdown
    word_details = process_words(ref_text_normalized, hyp_text_normalized)
    word_error_counts = {
        "Correct": word_details.hits,
        "Substitution": word_details.substitutions,
        "Deletion": word_details.deletions,
        "Insertion": word_details.insertions,
    }

    # Character-level error breakdown
    char_details = process_characters(ref_text_normalized, hyp_text_normalized)
    char_error_counts = {
        "Correct": char_details.hits,
        "Substitution": char_details.substitutions,
        "Deletion": char_details.deletions,
        "Insertion": char_details.insertions,
    }

    # ✅ NEW: Speaker similarity (voice cloning)
    speaker_similarity = None
    if ref_audio_path is not None:
        if speaker_sim_model is None:
            speaker_sim_model = get_speaker_sim_model()  # lazy init
        speaker_similarity = speaker_sim_model.cosine_similarity(str(ref_audio_path), str(audio_path))

    # Prepare result directory
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # Create JSON filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_name = Path(audio_path).stem
    json_filename = f"{model_name}.json"
    json_path = result_path / json_filename

    # Save results to JSON
    results_to_save = {
        "model_name": model_name,
        "audio_file": str(audio_path),
        "reference": ref_text,
        "hypothesis": hyp_text,
        "WER": wer_score,
        "CER": cer_score,
        "word_error_breakdown": word_error_counts,
        "char_error_breakdown": char_error_counts,
        "speaker_ref_audio": str(ref_audio_path) if ref_audio_path is not None else None,  # ✅ NEW
        "speaker_similarity_cosine": speaker_similarity,                                   # ✅ NEW
        "asr_model": asr_model_name,
        "timestamp": timestamp
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved metrics (WER+CER+SIM) to: {json_path}")
    return str(json_path)

def plot_comparison(json_paths, result_dir="../results", chart_filename=None):
    """
    Vẽ đồ thị so sánh WER + CER + Speaker Similarity (nếu có)
    """
    # Load all JSON data
    all_results = []
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.append(data)

    model_names = [r.get("model_name", "Unknown") for r in all_results]
    wer_scores = [r.get("WER", 0) * 100 for r in all_results]  # %
    cer_scores = [r.get("CER", 0) * 100 for r in all_results]  # %
    sim_scores = [r.get("speaker_similarity_cosine", None) for r in all_results]  # float or None

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.Set2(range(len(model_names)))

    # Plot 1: WER
    bars1 = ax1.bar(model_names, wer_scores, color=colors)
    ax1.set_xlabel("Model TTS")
    ax1.set_ylabel("WER (%)")
    ax1.set_title("So sánh Word Error Rate (WER)")
    ax1.set_ylim(0, max(wer_scores) * 1.2 if wer_scores else 10)
    for bar, score in zip(bars1, wer_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{score:.2f}%', ha='center', va='bottom', fontsize=10)

    # Plot 2: CER
    bars2 = ax2.bar(model_names, cer_scores, color=colors)
    ax2.set_xlabel("Model TTS")
    ax2.set_ylabel("CER (%)")
    ax2.set_title("So sánh Character Error Rate (CER)")
    ax2.set_ylim(0, max(cer_scores) * 1.2 if cer_scores else 10)
    for bar, score in zip(bars2, cer_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{score:.2f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Speaker Similarity
    # Nếu thiếu similarity (None) thì vẽ 0 và annotate "N/A"
    sim_plot_vals = [(s if s is not None else 0.0) for s in sim_scores]
    bars3 = ax3.bar(model_names, sim_plot_vals, color=colors)
    ax3.set_xlabel("Model TTS")
    ax3.set_ylabel("Cosine Similarity")
    ax3.set_title("So sánh Speaker Similarity (Voice Cloning)")
    ax3.set_ylim(0, 1.0)

    for bar, s in zip(bars3, sim_scores):
        label = f"{s:.3f}" if isinstance(s, (float, int)) else "N/A"
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save figure
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    if chart_filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_filename = f"metrics_comparison.png"

    chart_path = result_path / chart_filename
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved comparison chart to: {chart_path}")
    return str(chart_path)

def load_wer_results(json_path):
    """
    Đọc kết quả WER từ file JSON
    
    Args:
        json_path: str - đường dẫn đến file JSON
    
    Returns:
        dict - kết quả WER đã lưu
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)




# metric_calculate(
#     # audio_path="/home/lamquy/Project/TTS/results/F5-TTS-Vietnamese/20260116_143338/sample_1.wav", 
#     # audio_path="/home/lamquy/Project/TTS/results/XTTSv2/20260116_143925.wav", 
#     audio_path="/home/lamquy/Project/TTS/results/VieNeu-TTS/20260116_142158/sample_1.wav", 
#     ref_audio_path="/home/lamquy/Project/TTS/F5-TTS-Vietnamese/samples/khoi/khoi.wav",
#     ref_text="Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà.", 
#     model_name="VieNeu-TTS", 
#     asr_model_name="large", 
#     language="vi", 
#     result_dir="../results/json"
# )

plot_comparison([
    "/home/lamquy/Project/TTS/results/json/F5-TTS-Vietnamese_sample_1_metrics.json",
    "/home/lamquy/Project/TTS/results/json/VieNeu-TTS_sample_1_metrics.json",
    "/home/lamquy/Project/TTS/results/json/XTTSv2_20260116_143925_metrics.json"
])