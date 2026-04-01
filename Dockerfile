FROM python:3.11

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency file first so Docker can cache the install layer —
# dependencies are only reinstalled when pyproject.toml changes
COPY pyproject.toml .

# Install all dependencies into the system Python
RUN uv pip install --system --no-cache -e .

# Copy the rest of the project (scripts/, raw data, and model artifacts are excluded via .dockerignore)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
