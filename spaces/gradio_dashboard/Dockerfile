FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio pydantic anthropic openai httpx

COPY . /app

ENV PYTHONPATH="/app:/app/bench/gradio_app:$PYTHONPATH"
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860

CMD ["bash", "-c", "cd /app/bench/gradio_app && python app.py"]
