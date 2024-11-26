# Use the official Python image.
FROM python:3.8-slim

# Set the working directory.
WORKDIR /app

# Copy the requirements file.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code.
COPY . .

# Expose ports if necessary (e.g., for a web app).
# EXPOSE 8080

# Set environment variables if needed.
# ENV VARIABLE_NAME=value

# Set the entry point to your training script.
ENTRYPOINT ["python", "train.py"]
