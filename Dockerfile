FROM tensorflow/tensorflow:2.12.0

# Set a working directory for the build
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python packages required for your application
RUN pip install --no-cache-dir -r requirements.txt

# Set a working directory for the app
COPY . /app/

# Expose the port that your FastAPI app will run on
EXPOSE 8000

# Define the command to run your FastAPI app
CMD ["uvicorn", "src.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
