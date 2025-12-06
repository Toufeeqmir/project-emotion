FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir -p /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV MPLCONFIGDIR=/code/cache
RUN chmod -R 777 /code/cache
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]