FROM tensorflow/tensorflow:2.1.0-gpu-py3

COPY /home/llorenc/TrajectoryPrediction requirements.txt

RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && pip install jupyter \
    && pip install -r requirements.txt
