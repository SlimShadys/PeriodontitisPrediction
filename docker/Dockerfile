FROM python:3.12.9-bookworm

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt /

# Install Python dependencies
RUN pip install -r /requirements.txt

# Cache directory creation
RUN mkdir /.cache && chmod 777 /.cache

# Here we should enable WandB with: yolo settings wandb=True at '/root/.config/Ultralytics/settings.json'
RUN yolo settings wandb=True
