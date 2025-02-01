# peter-audio_file
Sequential version of the pipeline for file audio 

Tested on python 3.10.12 on Linux 

Since the script uses whisperX, it is necessary to have ffmpeg installed:
  sudo apt install ffmpeg

## To run the script:
1. Open the requirements.txt file and install the libraries one by one.
2. Run python3 sequential_version.py
3. The parameters that can be changed are within the main, they are: file to be translated, source language, target language, chunk size and speaker:
   * In speakers folder insert the audio file from which the base voice will be used for tts.
   * In audio_inputs the audio to be processed.
   * Chunk size reccommended: 5 or 3. 

## Outputs:
- The output_tts_chunks folder stores the translated audios for each chunk in wav format.
- The file execution_times.csv in the home folder contains the csv with the times recorded for that execution.

