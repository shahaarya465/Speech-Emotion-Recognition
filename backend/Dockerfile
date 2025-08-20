# Use the Python version that matches your environment
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy all files from the project root into the container
# This now includes app.py, requirements.txt, and the export/ folder
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 7860

# The command to run the app is now simpler because app.py is at the root
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]