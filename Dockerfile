FROM debian:bullseye
WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt update && \
    apt install python3-opencv freeglut3 python3-pip -y --no-install-recommends && \
    pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN mkdir data
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]