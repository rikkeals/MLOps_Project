# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts of the project to container
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/
# data is mounted at runtime to keep the image smaller

# Set working directory in container
WORKDIR /

# Install dependencies and packages
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir 
# The above does not work with Google Cloud Build yet, so we use the below as a workaround
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Install gsutil (Google Cloud SDK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*


# Entry point (what should be run when image is executed)
ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]