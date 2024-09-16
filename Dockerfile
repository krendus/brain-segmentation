# Use the official Python 3.10 image as the base image
FROM python:3.10.14-slim

# Set environment variables to avoid buffering of logs and ensure correct behavior of Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# Expose the port Flask will run on
EXPOSE 5002

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5002"]
