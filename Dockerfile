FROM ubuntu:18.04
#MAINTAINER Poojan "poojanthaker@gmail.com"
RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get install wget -y
RUN wget https://pjreddie.com/media/files/yolov3.weights
COPY . /app
ENTRYPOINT [ "python3 app.py"]