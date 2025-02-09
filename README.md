# peter-audio_file
Sequential version of the pipeline for file audio 


## To run the script:
1. Build the Dockerfile: sudo docker build -t my-whisperx-app .
2. Run the builded container (with gpu) :  sudo docker run --gpus all -it my-whisperx-app 
 

## Outputs:
- The final audio with all the chunk merged is final_output.wav
- The file execution_times.csv in the home folder contains the csv with the times recorded for that execution.

