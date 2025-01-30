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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    f0_mean_threshold = 270
    f0_std_threshold = 50
    pitch_range_threshold = 50
    #speech_rate_threshold = 100
    #articulation_rate_threshold = 120
    avg_pause_duration_threshold = 0.4
    avg_phrase_length_threshold = 2.0
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

def sequential_pipeline(file_path, src_lan, trg_lan):
    #caricamento modelli
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
    chunk_duration = 5  # Duration chunk
    chunk_overlap = 0.3  # Overlap between chunks
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(chunk_overlap * sample_rate)
    total_samples = audio.shape[1]

    #previous_transcription = ""

    for start in range(0, total_samples, chunk_samples - overlap_samples):

        end = min(start + chunk_samples, total_samples)
        audio_chunk = audio[:, start:end]

        logger.info(f"Compute chunk: {start/sample_rate:.2f} - {end/sample_rate:.2f} seconds")

        temp_chunk_path = "temp_chunk.wav"
        torchaudio.save(temp_chunk_path, audio_chunk, sample_rate)
        audio_chunk_np = whisperx.load_audio(temp_chunk_path)


        def classification_task(path):
                try:
                    #num_words = len(text.split())
                    features = extract_urgency_features(path)
                    classification = classify_urgency(features)
                    logger.info(f"Urgency classification: {classification}")
                    #os.remove(path)
                except Exception as e:
                    logger.error(f"Error in classification: {str(e)}")

        classification_thread = threading.Thread(
                target=classification_task,
                args=(temp_chunk_path,)
            )
        classification_thread.start()

        logger.info("Start transcription...")
        start_stt_time = time.time() #timer for stt
        result = whisper_model.transcribe(audio_chunk_np, batch_size=16, language=src_lan)
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
        translated = translation_model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translation_time = time.time() - start_translation_time
        logger.info(f"Translation completed: {translated_text}")
        logger.info(f"Time for translation: {translation_time:.2f} seconds")

        # Text to speech
        logger.info("Start text to speech...")
        chunk_index = start // (chunk_samples - overlap_samples)
        output_tts_path = f"output_tts_chunk_{chunk_index}.wav"
        speaker = "speaker/audio_ita.wav" #here to specify the base voice to use in tts
        start_tts_time = time.time()
        tts_model.tts_to_file(text=translated_text, file_path=output_tts_path, speaker_wav=speaker, language=trg_lan)
        tts_time = time.time() - start_tts_time
        logger.info(f"output tts chunk saved as:{output_tts_path}")
        logger.info(f"TTS completed in {tts_time:.2f} seconds.")

if __name__ == "__main__":
    audio_file = "audio_ita.wav"
    src_lan = "it"
    trg_lan = "en"
    sequential_pipeline(audio_file, src_lan, trg_lan)
