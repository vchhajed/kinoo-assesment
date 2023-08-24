# Choose a base image suitable for your project
FROM python:3.8

WORKDIR /app

# Install required dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project code into the container
COPY . .

# Set the entry point (modify this based on your use case)
CMD ["python", "src/train.py"]
