# Use an official Python image as the base
FROM python:3.9

# Install system dependencies and Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port (important for Render)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
