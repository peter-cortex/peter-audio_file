import torch
import whisperx
import torchaudio
import numpy as np
import time
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import os
import logging
import librosa
import parselmouth
from parselmouth.praat import call
import threading
import csv
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SpecificFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "Compute chunk" in message or "Urgency classification" in message:
            return True
        return False

file_handler = logging.FileHandler("process_logs.txt", mode='w')
file_handler.setLevel(logging.INFO)
file_handler.addFilter(SpecificFilter())
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

CSV_FILE = "execution_times.csv"
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Chunk", "STT Time (s)", "Translation Time (s)", "TTS Time (s)", "Urgency Classification Time (s)", "Total processing time (s)"])

def save_execution_time(chunk_index, stt_time, translation_time, tts_time, urgency_time):
    total_processing_time = round(stt_time + translation_time + tts_time, 3)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([chunk_index, stt_time, translation_time, tts_time, urgency_time, total_processing_time])

def append_statistics_to_csv():
    df = pd.read_csv(CSV_FILE)
    avg_values = df.iloc[:, 1:6].mean().round(3).tolist()
    std_values = df.iloc[:, 1:6].std().round(3).tolist()
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["AVG"] + avg_values)
        writer.writerow(["STD"] + std_values)

def calculate_phrase_lengths(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    phrase_lengths = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    return phrase_lengths

def extract_urgency_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    speaking_time = sum([(end - start) / sr for start, end in non_silent_intervals])
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch()
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    pause_durations = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    num_pauses = len(pause_durations)
    phrase_lengths = calculate_phrase_lengths(audio_file)
    avg_phrase_length = np.mean(phrase_lengths) if phrase_lengths else 0
    pitch_range = f0_max - f0_min
    rms_amplitude = librosa.feature.rms(y=y).mean()
    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "pitch_range": pitch_range,
        "avg_pause_duration": avg_pause_duration,
        "num_pauses": num_pauses,
        "avg_phrase_length": avg_phrase_length,
        "rms_amplitude": rms_amplitude,
        "speaking_time": speaking_time,
        "duration": duration,
    }

def classify_urgency(features):
    f0_mean_threshold = 210
    f0_std_threshold = 40
    pitch_range_threshold = 50
    avg_pause_duration_threshold = 0.4
    avg_phrase_length_threshold = 3.0
    rms_amplitude_threshold = 0.1
    is_urgent = (
        features["f0_mean"] > f0_mean_threshold and
        features["f0_std"] > f0_std_threshold and
        features["pitch_range"] > pitch_range_threshold and
        features["avg_pause_duration"] < avg_pause_duration_threshold and
        features["avg_phrase_length"] < avg_phrase_length_threshold and
        features["rms_amplitude"] > rms_amplitude_threshold
    )
    return "URGENT" if is_urgent else "NOT URGENT"

def apply_crossfade(chunk1, chunk2, crossfade_duration, sample_rate):
    n_samples = int(crossfade_duration * sample_rate)
    if chunk1.shape[1] < n_samples or chunk2.shape[1] < n_samples:
        return torch.cat((chunk1, chunk2), dim=1)
    fade_out = torch.linspace(1.0, 0.0, n_samples).to(chunk1.device)
    fade_in = torch.linspace(0.0, 1.0, n_samples).to(chunk2.device)
    chunk1_end = chunk1[:, -n_samples:] * fade_out
    chunk2_start = chunk2[:, :n_samples] * fade_in
    crossfaded = chunk1_end + chunk2_start
    merged = torch.cat((chunk1[:, :-n_samples], crossfaded, chunk2[:, n_samples:]), dim=1)
    return merged

def create_chunk(audio, sample_rate, start, end, chunk_index, temp_dir="output_temp"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    chunk_path = os.path.join(temp_dir, f"temp_chunk_{chunk_index}.wav")
    audio_chunk = audio[:, start:end]
    torchaudio.save(chunk_path, audio_chunk, sample_rate)

    urgency_result = {}
    def classification_task(path, result_dict, idx):
        try:
            start_time = time.time()
            features = extract_urgency_features(path)
            classification = classify_urgency(features)
            result_dict["urgency_time"] = time.time() - start_time
            result_dict["classification"] = classification
        except Exception as e:
            logger.error(f"Error in classification for chunk {idx}: {str(e)}")
            result_dict["urgency_time"] = 0.0
            result_dict["classification"] = "NOT URGENT"

    thread = threading.Thread(target=classification_task, args=(chunk_path, urgency_result, chunk_index))
    thread.start()
    return chunk_path, thread, urgency_result

def stt_phase(whisper_model, align_model, align_metadata, chunk_path, device, src_lan):
    logger.info("Start transcription (STT)...")
    start_time = time.time()
    audio_chunk_np = whisperx.load_audio(chunk_path)
    result = whisper_model.transcribe(audio_chunk_np, batch_size=16, language=src_lan)
    if not result["segments"]:
        logger.info("No active speech found in chunk.")
        return "", 0.0
    aligned_result = whisperx.align(result["segments"], align_model, align_metadata, chunk_path, device)
    transcription = " ".join(segment['text'] for segment in aligned_result["segments"])
    stt_time = time.time() - start_time
    logger.info(f"Transcription: {transcription}")
    logger.info(f"Time for STT: {stt_time:.2f} seconds")
    return transcription, stt_time

def translation_phase(translation_model, tokenizer, text, src_lan, trg_lan, device):
    logger.info("Start translation...")
    start_time = time.time()
    src_text = f">>{src_lan}<< {text.strip()}"
    translated = translation_model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device), do_sample=False)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    if translated_text:
        translated_text = translated_text.replace('.', ',')
    if translated_text.strip() == "==References====External links==":
        logger.warning("Translation produced a default placeholder. Using original transcription as fallback.")
        translated_text = text.strip()
    translation_time = time.time() - start_time
    logger.info(f"Translated text: {translated_text}")
    logger.info(f"Time for translation: {translation_time:.2f} seconds")
    return translated_text, translation_time

def tts_phase(tts_model, text, trg_lan, classification, chunk_index, output_dir="output_tts_chunk"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"output_tts_chunk_{chunk_index}.wav")
    logger.info("Start text-to-speech (TTS)...")
    start_time = time.time()
    multi_speaker_urgent = ["speakers/my_urgent_audio_1.wav", "speakers/my_urgent_audio_2.wav"]
    multi_speaker_not_urgent = ["speakers/my_not_urgent_audio_1.wav", "speakers/my_not_urgent_audio_2.wav"]
    speaker_wavs = multi_speaker_urgent if classification == "URGENT" else multi_speaker_not_urgent
    tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wavs, language=trg_lan)
    tts_time = time.time() - start_time
    logger.info(f"TTS output saved: {output_path}")
    logger.info(f"Time for TTS: {tts_time:.2f} seconds")
    return output_path, tts_time

def sequential_pipeline(file_path, src_lan, trg_lan, chunk_duration):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device}")
    compute_type = "float32"

    logger.info("Loading WhisperX model...")
    whisper_model = whisperx.load_model("turbo", device=device, compute_type=compute_type, language=src_lan)
    align_model, align_metadata = whisperx.load_align_model(language_code=src_lan, device=device)

    logger.info("Loading translation model...")
    translation_model_name = f"Helsinki-NLP/opus-mt-{src_lan}-{trg_lan}"
    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    logger.info("Loading Coqui TTS model...")
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    audio, sample_rate = torchaudio.load(file_path)
    chunk_overlap = 0.1
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(chunk_overlap * sample_rate)
    total_samples = audio.shape[1]
    tts_audio_chunks = []
    tts_output_sample_rate = None
    crossfade_duration = 0.2

    chunk_index = 0
    for start in range(0, total_samples, chunk_samples - overlap_samples):
        end = min(start + chunk_samples, total_samples)
        
        chunk_path, classification_thread, urgency_result = create_chunk(audio, sample_rate, start, end, chunk_index)

        transcription, stt_time = stt_phase(whisper_model, align_model, align_metadata, chunk_path, device, src_lan)
        if not transcription:
            chunk_index += 1
            continue

       
        translated_text, translation_time = translation_phase(translation_model, tokenizer, transcription, src_lan, trg_lan, device)

        classification_thread.join()
        urgency_time = urgency_result.get("urgency_time", 0.0)
        classification = urgency_result.get("classification", "NOT URGENT")

        output_tts_path, tts_time = tts_phase(tts_model, translated_text, trg_lan, classification, chunk_index)

        save_execution_time(chunk_index, round(stt_time, 2), round(translation_time, 2), round(tts_time, 2), round(urgency_time, 3))

        try:
            tts_waveform, tts_sr = torchaudio.load(output_tts_path)
            if tts_output_sample_rate is None:
                tts_output_sample_rate = tts_sr
            tts_audio_chunks.append(tts_waveform)
        except Exception as e:
            logger.error(f"Error loading TTS chunk audio {output_tts_path}: {str(e)}")

        chunk_index += 1

    append_statistics_to_csv()

    if tts_audio_chunks:
        final_audio = tts_audio_chunks[0]
        for chunk in tts_audio_chunks[1:]:
            final_audio = apply_crossfade(final_audio, chunk, crossfade_duration, tts_output_sample_rate)
        final_output_path = "final_output.wav"
        torchaudio.save(final_output_path, final_audio, tts_output_sample_rate)
        logger.info(f"Final merged TTS audio saved as: {final_output_path}")
    else:
        logger.info("No TTS chunk generated, final file not created.")

if __name__ == "__main__":
    audio_file = "audio_en.wav"
    chunk_duration = 5
    src_lan = "en"
    trg_lan = "it"
    #sequential_pipeline(audio_file, src_lan, trg_lan, chunk_duration)
    parser = argparse.ArgumentParser(description="Esegui la pipeline STT/Translation/TTS")
    parser.add_argument("--audio_file", type=str, default="audio_en.wav", help="Audio file path")
    parser.add_argument("--src", type=str, default="en", help="Source Language (es. 'en, fr, de')")
    parser.add_argument("--trg", type=str, default="it", help="Destination Language (es. 'en, fr, de')")
    parser.add_argument("--chunk_duration", type=int, default=5, help="Time for each chunk (suggested: 5 seconds)")
    args = parser.parse_args()
    #sequential_pipeline(audio_file, src_lan, trg_lan, chunk_duration)
    sequential_pipeline(args.audio_file, args.src, args.trg, args.chunk_duration)
