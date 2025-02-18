# peter-audio_file
Sequential version of the pipeline for file audio 


## To run the script:
1. Build the Dockerfile:
```bash
sudo docker build -t my-peter-app .
```
4. Run the builded container (with gpu) :
```bash
sudo docker run --gpus all -it my-peter-app
```
6. To run the script with the parameters and personalized audio file:  
```bash
sudo docker run --gpus all -it -v $(pwd)/file_audio.wav:/app/file_audio.wav my-peter-app --audio_file /app/audio_en.wav --src en --trg fr --chunk_duration 5
```
 

## Outputs:
- The final audio with all the chunk merged is final_output.wav
- The file execution_times.csv in the home folder contains the csv with the times recorded for that execution.

