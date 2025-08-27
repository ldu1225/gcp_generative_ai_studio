# Use the official full Python image to ensure all system dependencies are available.
# https://hub.docker.com/_/python
FROM python:3.12

# Allow statements and log messages to be sent straight to the terminal.
ENV PYTHONUNBUFFERED True

# Install system dependencies, including the font required by the application.
RUN apt-get update && apt-get install -y fonts-dejavu-core

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Set the working directory to the backend folder
WORKDIR /app/backend

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Expose the port that the application listens on. Cloud Run will use the value of PORT.
EXPOSE 8080

# Run the application directly using the Flask development server.
# This mirrors the local execution environment.
CMD ["python", "main.py"]
