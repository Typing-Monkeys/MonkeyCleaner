FROM tensorflow/tensorflow:2.2.1-gpu-py3
MAINTAINER ncvescera nicolo.vescera@gmail.com

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install pygame
RUN pip install opencv-python
RUN pip install keras
RUN pip install -U scikit-learn
