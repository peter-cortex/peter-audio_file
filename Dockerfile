FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    gnupg \
    ca-certificates \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

#CUDA/CUDNN installation
RUN apt-get update && apt-get install -y wget gnupg lsb-release && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 3bf863cc.pub && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
         > /etc/apt/sources.list.d/cuda.list && \
    rm 3bf863cc.pub && \
    apt-get update && apt-get install -y libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

#Download of whisperX and coqui ai TTS
ENV TORCH_HOME=/app/models
RUN python -c "import whisperx; _ = whisperx.load_model('turbo', device='cpu', compute_type='float32')"
RUN echo "y" | python -c "from TTS.api import TTS; _ = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)"


COPY sequential_version.py /app/
COPY speakers /app/speakers
COPY speakers/audio_en.wav /app/
COPY speakers/my_urgent_audio_1.wav /app/
COPY speakers/my_urgent_audio_2.wav /app/
COPY speakers/my_not_urgent_audio_1.wav /app/
COPY speakers/my_not_urgent_audio_2.wav /app/
COPY wer_compute.py /app/
COPY wer_audio /app/wer_audio

ENV COQUI_TOS_AGREED=1

ENTRYPOINT ["python", "peter_file.py"]
CMD []
