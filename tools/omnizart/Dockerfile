FROM python:3.8-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "numpy<2" "Cython<3"

RUN pip install --no-cache-dir --no-build-isolation "madmom==0.16.1"
RUN pip install --no-cache-dir omnizart

WORKDIR /work

CMD ["omnizart", "--help"]
