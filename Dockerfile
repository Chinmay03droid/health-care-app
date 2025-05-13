FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Optional: create a virtual environment
# RUN python -m venv venv
# ENV PATH="/app/venv/bin:$PATH"

# Copy and install dependencies early for caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /app/

# Optional: use a non-root user
# RUN adduser --disabled-password --gecos '' appuser
# USER appuser

EXPOSE 5000

CMD ["python", "health-care-app.py"]

