# Use a base image suitable for your application
FROM python:3.8

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app/ .

EXPOSE 5000

# Run the Flask app
CMD ["python", "main.py"]
