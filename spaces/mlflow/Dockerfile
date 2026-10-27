FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user basic_auth.ini.template .
COPY --chown=user start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
