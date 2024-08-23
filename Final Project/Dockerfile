FROM python:3.11.0-slim

# Install the necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Create a working directory
RUN mkdir app
WORKDIR /app

# Copy the application files
COPY . /app

# Install the Python dependencies
RUN pip install -r requirements.txt

# Expose the port 
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "8000"]
