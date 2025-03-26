FROM python:3.12.9-bookworm

RUN apt update && apt install -y build-essential

COPY requirements.txt /

# RUN pip3 install PACCHETTI_FONDAMENTALI

RUN pip install -r /requirements.txt

RUN mkdir /.cache && chmod 777 /.cache
