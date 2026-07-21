FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY sommelier ./sommelier
COPY scripts ./scripts
COPY llm_module.py ./llm_module.py

RUN pip install --upgrade pip \
    && pip install ".[vector]"

COPY data ./data

EXPOSE 8012

CMD ["python", "-m", "uvicorn", "sommelier.web.app:app", "--host", "0.0.0.0", "--port", "8012"]
