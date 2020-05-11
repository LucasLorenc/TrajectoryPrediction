#!/bin/bash

docker build -f docker/Dockerfile -t train-tf21 .
docker run -it --name tf21 --gpus all -w /home/llorenc/TrajectoryPrediction \
	--volume /home/llorenc/disk_1TB/data:/home/llorenc/TrajectoryPrediction/data \
	-p 8888:8888 -w /home/llorenc/TrajectoryPrediction train-tf21:latest \
        bash
