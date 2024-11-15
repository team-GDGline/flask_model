# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Prevent interactive dialog during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install Python and necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
        python3-pip \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default Python version
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable to disable buffering, helpful for logging
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
