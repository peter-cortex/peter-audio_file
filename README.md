# peter-audio_file
Sequential version of the pipeline for file audio 


## To run the script:
1. Build the Dockerfile:
```bash
sudo docker build -t my-peter-app .
```
2. Run the builded container (with gpu) :
```bash
 sudo docker run --gpus all -it -v "$(pwd)/output:/app/output" my-peter-app
```
3. To run the script with the parameters and personalized audio file:  
```bash
 sudo docker run --gpus all -it     -v "$(pwd)/audio_input:/app/audio_input"     -v "$(pwd)/output:/app/output"     my-peter-app     python /app/sequential_version.py --audio_file /app/audio_input/audio_fr.wav --src fr --trg en --chunk_duration 5
```
The input audio must be placed inside the audio_input folder
 

## Outputs:
- The final audio with all the chunk merged is output/final_output.wav

## Possible errors in building phase:
If the building stops at FROM ubuntu:22.04, it is necessary to download the image from Docker:
```bash
docker pull ubuntu:22.04
```
If the building stops at libcudnn 8 not found probably CUDA or cuDNN are not installed, please do so. 


## Evaluate metrics
There are scripts for the evaluation of certain statistics:
- The WER (Word Error Rate) measures erroneous words on audio transcripts. If you want to use your own audio files for evaluation, they must be loaded into the wer_audio folder, in the same way as the other folders already present (audio + transcription file). Use the following command to run the script, the results will be in the usual output folder:
```bash
sudo docker run --gpus all -it -v "$(pwd)/output:/app/output" --entrypoint python my-peter-app wer_compute.py
```

