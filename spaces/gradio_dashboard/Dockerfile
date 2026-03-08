FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio pydantic anthropic openai

COPY . /app

ENV PYTHONPATH="/app:/app/bench/gradio_app:$PYTHONPATH"

EXPOSE 7860

CMD ["bash", "-c", "cd /app/bench/gradio_app && python app.py"]
