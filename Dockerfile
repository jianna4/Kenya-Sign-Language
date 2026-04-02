FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# ==============================
# SYSTEM DEPENDENCIES

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 \
    libglib2.0-0 libgomp1 libxcb-shm0 libxcb-xfixes0 \
    libxcb-randr0 curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install


# NODE (for Gradio)
# ==============================
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*


# WORKDIR

WORKDIR /app


# INSTALL PYTHON DEPENDENCIES

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================
# COPY EVERYTHING (🔥 FIX HERE)
# ==============================
COPY . /app


# PORT

EXPOSE 7860

# RUN APP

CMD ["python", "app.py"]