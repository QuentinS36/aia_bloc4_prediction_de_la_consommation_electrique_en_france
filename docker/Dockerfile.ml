FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements/base.txt requirements/ml.txt requirements/
RUN pip install -r requirements/ml.txt
COPY dday dday
RUN useradd -m app && chown -R app /app
USER app
CMD ["python", "-m", "dday.ml.train"]
