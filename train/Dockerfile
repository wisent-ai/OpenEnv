FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Default: train with Qwen2.5-7B-Instruct, 500 steps
CMD ["python", "train.py", \
     "--model", "Qwen/Qwen2.5-7B-Instruct", \
     "--episodes", "2000", \
     "--max-steps", "500", \
     "--num-generations", "8", \
     "--batch-size", "2", \
     "--grad-accum", "8", \
     "--output-dir", "/workspace/output"]
