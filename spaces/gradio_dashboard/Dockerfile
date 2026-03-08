FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio pydantic anthropic openai

COPY . /app

ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

CMD ["python", "bench/gradio_app/app.py"]
