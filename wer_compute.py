import os
import glob
import jiwer
import torch
import torchaudio
import whisperx
import argparse

def load_transcriptions(trans_file):
    references = {}
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                audio_id, text = parts
                references[audio_id] = text
    return references

def create_chunk(audio, sample_rate, start, end, chunk_index, temp_dir="temp_chunks"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    chunk_path = os.path.join(temp_dir, f"temp_chunk_{chunk_index}.wav")
    audio_chunk = audio[:, start:end]
    torchaudio.save(chunk_path, audio_chunk, sample_rate)
    return chunk_path

def stt_phase(whisper_model, align_model, align_metadata, chunk_path, device, language):
    audio_chunk_np = whisperx.load_audio(chunk_path)
    result = whisper_model.transcribe(audio_chunk_np, batch_size=16, language=language)
    if not result["segments"]:
        return ""
    aligned_result = whisperx.align(
        result["segments"], align_model, align_metadata, chunk_path, device
    )
    transcription = " ".join(segment['text'] for segment in aligned_result["segments"])
    return transcription

def sequential_pipeline(file_path, whisper_model, align_model, align_metadata, device, language="en", chunk_duration=5.0):
    audio, sample_rate = torchaudio.load(file_path)
    chunk_overlap = 0.1
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(chunk_overlap * sample_rate)
    total_samples = audio.shape[1]
    all_chunk_transcriptions = []
    chunk_index = 0
    for start in range(0, total_samples, chunk_samples - overlap_samples):
        end = min(start + chunk_samples, total_samples)
        chunk_path = create_chunk(audio, sample_rate, start, end, chunk_index)
        transcription = stt_phase(whisper_model, align_model, align_metadata, chunk_path, device, language)
        if transcription:
            all_chunk_transcriptions.append(transcription)
        chunk_index += 1
    return " ".join(all_chunk_transcriptions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/app/wer_audio" ,required=False, help="Path to the main directory containing subdirectories with audio files and transcription files")
    args = parser.parse_args()
    main_directory = args.input_dir
    wer_output_file = "/app/output/wer_results.txt"
    transcriptions_output_file = "/app/output/transcriptions.txt"
    #transcriptions_output_file = os.path.join(main_directory, "transcriptions.txt")
    with open(wer_output_file, "w", encoding="utf-8") as f:
        f.write("=== WER RESULTS ===\n")
    with open(transcriptions_output_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED TRANSCRIPTIONS ===\n\n")
    wer_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32"
    print("Loading WhisperX and Pyannote models...")
    whisper_model = whisperx.load_model("turbo", device=device, compute_type=compute_type, language="en")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
    print("Models loaded successfully.")
    for sub_dir in sorted(os.listdir(main_directory)):
        sub_path = os.path.join(main_directory, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        trans_file = glob.glob(os.path.join(sub_path, "*.trans.txt"))
        if not trans_file:
            print(f"No transcription file found in {sub_path}")
            continue
        references = load_transcriptions(trans_file[0])
        audio_files = sorted(glob.glob(os.path.join(sub_path, "*.flac")))
        for audio_file in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            print(f"Processing file: {audio_file}")
            generated_transcription = sequential_pipeline(
                file_path=audio_file,
                whisper_model=whisper_model,
                align_model=align_model,
                align_metadata=align_metadata,
                device=device,
                language="en",
                chunk_duration=5.0
            )
            reference_text = references.get(base_name, "").strip()
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces()
            ])
            reference_text = transformation(reference_text).strip()
            generated_transcription = transformation(generated_transcription).strip()
            file_wer = jiwer.wer(reference_text, generated_transcription)
            wer_list.append(file_wer)
            with open(wer_output_file, "a", encoding="utf-8") as f:
                f.write(f"{audio_file} | WER: {file_wer:.3f}\n")
            with open(transcriptions_output_file, "a", encoding="utf-8") as f:
                f.write(f"File: {audio_file}\n")
                f.write(f"Generated Transcription:\n{generated_transcription}\n")
                f.write(f"Reference Transcription:\n{reference_text}\n")
                f.write("-" * 80 + "\n")
    if wer_list:
        average_wer = sum(wer_list) / len(wer_list)
    else:
        average_wer = 0.0
    with open(wer_output_file, "a", encoding="utf-8") as f:
        f.write(f"\nAverage WER over {len(wer_list)} files: {average_wer:.3f}\n")
    print(f"\n=== FINAL RESULTS ===")
    print(f"Average WER: {average_wer:.3f}")
    print(f"Details saved in: {wer_output_file}")

