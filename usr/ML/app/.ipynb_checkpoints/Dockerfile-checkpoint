FROM continuumio/anaconda3
COPY . /usr/ML/app
EXPOSE 5000
WORKDIR /usr/ML/app
RUN pip3 install -r requirements.txt --no-cache
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 -y
CMD python main.py

