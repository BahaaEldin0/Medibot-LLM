# Use the official Python base image
FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .



EXPOSE 5001
# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]



