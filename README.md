# peter-audio_file
Sequential version of the pipeline for file audio 


## To run the script:
1. Build the Dockerfile:
```bash
sudo docker build -t my-peter-app .
```
2. Run the builded container (with gpu) :
```bash
sudo docker run --gpus all -it my-peter-app
```
3. To run the script with the parameters and personalized audio file:  
```bash
 sudo docker run --gpus all -it -v "$(pwd)/file_audio:/app/audio_input" -v "$(pwd)/output:/app/output" my-peter-app --audio_file /app/audio_input/audio_en.wav --src en --trg fr --chunk_duration 5
```
 

## Outputs:
- The final audio with all the chunk merged is output/final_output.wav
- The file execution_times.csv in the home folder contains the csv with the times recorded for that execution.

## Possible errors in building phase:
If the building stops at FROM ubuntu:22.04, it is necessary to download the image from Docker:
```bash
docker pull ubuntu:22.04
```
If the building stops at libcudnn 8 not found probably CUDA or cuDNN are not installed, please do so. 

