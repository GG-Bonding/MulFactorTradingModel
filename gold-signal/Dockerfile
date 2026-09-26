FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY examples ./examples

ENV PYTHONPATH=/app/src
ENV AGENT_DB=/data/agents.sqlite
ENV HOST=0.0.0.0
ENV PORT=8765

VOLUME /data
EXPOSE 8765
CMD ["python", "-m", "gold_signal.api"]
