FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        fluidsynth \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Omnizart/madmom を通すため、NumPy/Cython を先に固定
RUN pip install --upgrade pip setuptools wheel \
    && pip install "numpy<2" "Cython<3" \
    && pip install --no-build-isolation "madmom==0.16.1" \
    && pip install -r requirements.txt \
    && pip install omnizart

COPY . /app

CMD ["python", "midimorph.py"]
