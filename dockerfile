# Use a slim official Python runtime image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port used by your UI (adjust if other port)
EXPOSE 7860

# Set environment variables if needed
ENV CUDA_LAUNCH_BLOCKING=1

# Default command to run app (adjust script name if different)
CMD ["python", "app.py"]
