# Use Python 3.9
FROM python:3.9

# Set working directory to /code
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a writable directory for caching (Fixes permission errors on HF)
RUN mkdir -p /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV MPLCONFIGDIR=/code/cache
RUN chmod -R 777 /code/cache

# Copy the rest of the application
COPY . .

# Run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
