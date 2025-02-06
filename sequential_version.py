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
    writer.writerow(["Chunk", "STT Time (s)", "Translation Time (s)", "TTS Time (s)", "Urgency Classification Time (parallel)", "Total processing time (s)",])
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
    #compute phrase lenght (a pharese are word between silence pause)
    y, sr = librosa.load(audio_file, sr=16000)
    non_silent_intervals = librosa.effects.split(y, top_db=30)

    phrase_lengths = [
        (non_silent_intervals[i][0] - non_silent_intervals[i - 1][1]) / sr
        for i in range(1, len(non_silent_intervals))
    ]
    return phrase_lengths

def extract_urgency_features(audio_file):
    #extraction of prosody features
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

    #speech_rate = num_words / (duration / 60) if duration > 0 else 0
    #articulation_rate = num_words / (speaking_time / 60) if speaking_time > 0 else 0

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
        #"speech_rate": speech_rate,
        #"articulation_rate": articulation_rate,
        "avg_pause_duration": avg_pause_duration,
        "num_pauses": num_pauses,
        "avg_phrase_length": avg_phrase_length,
        "rms_amplitude": rms_amplitude,
        "speaking_time": speaking_time,
        "duration": duration,
    }

def classify_urgency(features):
    """
    Classifica l'urgenza di un audio basandosi sulle feature estratte.
    """
    f0_mean_threshold = 210
    f0_std_threshold = 40
    pitch_range_threshold = 50
    #speech_rate_threshold = 100
    #articulation_rate_threshold = 120
    avg_pause_duration_threshold = 0.4
    avg_phrase_length_threshold = 3.0
    rms_amplitude_threshold = 0.1

    is_urgent = (
        features["f0_mean"] > f0_mean_threshold and
        features["f0_std"] > f0_std_threshold and
        features["pitch_range"] > pitch_range_threshold and
        #features["speech_rate"] > speech_rate_threshold and
        #features["articulation_rate"] > articulation_rate_threshold and
        features["avg_pause_duration"] < avg_pause_duration_threshold and
        features["avg_phrase_length"] < avg_phrase_length_threshold and
        features["rms_amplitude"] > rms_amplitude_threshold
    )

    return "URGENT" if is_urgent else "NOT URGENT"


def apply_crossfade(chunk1, chunk2, crossfade_duration, sample_rate):
    """
    Unisce due chunk audio applicando un crossfade di crossfade_duration secondi.
    chunk1 e chunk2 sono tensori di forma (channels, samples).
    """
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


def sequential_pipeline(file_path, src_lan, trg_lan, chunk_duration, speaker_voice):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32"

    logger.info("Caricamento del modello WhisperX...")
    whisper_model = whisperx.load_model(
        "turbo", device=device, compute_type=compute_type, language=src_lan
    )

    align_model, align_metadata = whisperx.load_align_model(language_code=src_lan, device=device)

    logger.info("Caricamento del modello di traduzione...")
    translation_model_name = f"Helsinki-NLP/opus-mt-{src_lan}-{trg_lan}"
    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    logger.info("Caricamento del modello Coqui TTS...")
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    audio, sample_rate = torchaudio.load(file_path)
    chunk_overlap = 0.1  # Overlap between chunks
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(chunk_overlap * sample_rate)
    total_samples = audio.shape[1]
    tts_audio_chunks = []
    tts_output_sample_rate = None
    crossfade_duration = 0.2

    #previous_transcription = ""

    for start in range(0, total_samples, chunk_samples - overlap_samples):

        end = min(start + chunk_samples, total_samples)
        audio_chunk = audio[:, start:end]

        logger.info(f"Compute chunk: {start/sample_rate:.2f} - {end/sample_rate:.2f} seconds")

        temp_chunk_path = "output_temp/temp_chunk.wav"
        torchaudio.save(temp_chunk_path, audio_chunk, sample_rate)
        audio_chunk_np = whisperx.load_audio(temp_chunk_path)
        chunk_index = start // (chunk_samples - overlap_samples)
        result_dict = {}

        def classification_task(path, result_dict, chunk_index):
                try:
                    #num_words = len(text.split())
                    start_urgency_time = time.time()
                    features = extract_urgency_features(path)
                    logger.info(f"fundamental frequency for chunk {chunk_index}: {features}")
                    classification = classify_urgency(features)
                    logger.info(f"Urgency classification: {classification}")
                    urgency_time = time.time() - start_urgency_time
                    logger.info(f"Urgency classified in: {urgency_time:.3f} for chunk {chunk_index}")
                    result_dict[chunk_index] = {"urgency_time": urgency_time, "classification": classification}
                    #os.remove(path)
                except Exception as e:
                    logger.error(f"Error in classification: {str(e)}")
                    result_dict[chunk_index] = {"urgency_time": 0.0, "classification": "NOT URGENT"}

        classification_thread = threading.Thread(
                target=classification_task,
                args=(temp_chunk_path, result_dict, chunk_index)
            )
        classification_thread.start()



        logger.info("Start transcription...")
        start_stt_time = time.time() #timer for stt
        result = whisper_model.transcribe(audio_chunk_np, batch_size=16, language=src_lan)
        if not result["segments"]:
            logger.info("No active speech found in audio, skipping chunk.")
            continue
        aligned_result = whisperx.align(result["segments"], align_model, align_metadata, temp_chunk_path, device)
        current_transcription = " ".join(segment['text'] for segment in aligned_result["segments"])
        logger.info(f"text: {current_transcription}")

        #previous_transcription = current_transcription.strip()
        transcription_time = time.time() - start_stt_time
        logger.info(f"Time for transcription: {transcription_time:.2f} secondi")

        # Translation
        logger.info("Start translation...")
        start_translation_time = time.time()
        src_text = f">>{src_lan}<< {current_transcription.strip()}"
        #print(f"testo per la traduzione {src_text}")
        #src_text = current_transcription
        translated = translation_model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device), do_sample=False)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        if translated_text and translated_text.strip().endswith('.'):
          translated_text = translated_text.strip()[:-1]
        if translated_text.strip() == "==References====External links==":
          logger.warning("Translation produced a default placeholder. Using the original transcription as fallback.")
          translated_text = current_transcription.strip()
        translation_time = time.time() - start_translation_time
        logger.info(f"Translation completed: {translated_text}")
        logger.info(f"Time for translation: {translation_time:.2f} seconds")

        #retrieve classification urgency
        classification_thread.join()
        #urgency_time = result_dict.get(chunk_index, 0.0)
        urgency_result = result_dict.get(chunk_index, {"urgency_time": 0.0, "classification": "NOT URGENT"})
        urgency_time = urgency_result["urgency_time"]
        classification = urgency_result["classification"]


        # Text to speech
        logger.info("Start text to speech...")

        output_tts_path = f"output_tts_chunk_fr_en/output_tts_chunk_{chunk_index}.wav"
        #speaker = "audio_ita.wav" #here to specify the base voice to use in tts
        start_tts_time = time.time()
        #urgent_text = "["+translated_text+"!]"
        multi_speaker_urgent = ["my_urgent_audio_1.wav", "my_urgent_audio_2.wav"]
        multi_speaker_not_urgent = ["my_not_urgent_audio_1.wav", "my_not_urgent_audio_2.wav"]
        speaker_wavs = multi_speaker_urgent if classification == "URGENT" else multi_speaker_not_urgent
        tts_model.tts_to_file(text=translated_text, file_path=output_tts_path, speaker_wav=speaker_wavs, language=trg_lan)
        tts_time = time.time() - start_tts_time
        logger.info(f"output tts chunk saved as:{output_tts_path}")
        logger.info(f"TTS completed in {tts_time:.2f} seconds.")


        save_execution_time(chunk_index, round(transcription_time, 2) , round(translation_time, 2), round(tts_time, 2), round(urgency_time, 3))

        try:
            tts_waveform, tts_sr = torchaudio.load(output_tts_path)
            if tts_output_sample_rate is None:
                tts_output_sample_rate = tts_sr
            tts_audio_chunks.append(tts_waveform)
            #channels = tts_waveform.shape[0]
            #modified_channels = []
            #rate = 1.2
            #for i in range(channels):
             # y = tts_waveform[i].numpy()
              # Applica il time stretch
             # y = librosa.effects.time_stretch(y, rate=rate)
              #y = librosa.effects.pitch_shift(y, sr=tts_sr, n_steps=2)
             # y *= 1.5
              # Converti di nuovo in tensore
             # modified_channels.append(torch.from_numpy(y))
            # Ricompone i canali: supponendo che ogni canale abbia ora la stessa lunghezza
          #  modified_waveform = torch.stack(modified_channels)
          #  tts_audio_chunks.append(modified_waveform)
        except Exception as e:
            logger.error(f"Error loading TTS chunk audio {output_tts_path}: {str(e)}")


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
    audio_file = "ita_emergenza.wav"
    #speaker_voice = "audio_en_21.wav"
    chunk_duration = 5
    src_lan = "it"
    trg_lan = "en"
    match trg_lan:
        case "it":
            speaker_voice = "speakers/audio_ita.wav"
        case "fr":
            speaker_voice = "speakers/audio_fr2.wav"
        case "de":
            speaker_voice = "speakers/audio_de_10.wav"
        case "en":
            speaker_voice = "speakers/audio_en.wav"
        case _:
            speaker_voice = "speakers/audio_en.wav"
    #speaker_voice = "em_audio_en.wav"
    sequential_pipeline(audio_file, src_lan, trg_lan, chunk_duration, speaker_voice)
