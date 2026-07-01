FROM debian:bookworm-slim

# libgomp1: OpenMP runtime needed by scikit-learn and OpenFHE native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    git \
    ca-certificates

# Build and install OpenFHE
RUN git clone https://github.com/openfheorg/openfhe-development
WORKDIR /openfhe-development
RUN cmake -BBuild .
WORKDIR Build
RUN cmake --build . -- -j$(nproc)
RUN cmake --install .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml .

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN uv sync

COPY . .

# The Shanghai T2DM dataset cannot be shipped in this image (licensed data).
# Mount the directory of .xls/.xlsx files at runtime:
#   docker run -v /path/to/Shanghai_T2DM:/app/dataset/shanghai_t2dm <image>
VOLUME ["/app/dataset/shanghai_t2dm"]

RUN mkdir -p /app/src/models/nn/checkpoints /app/src/models/tree/checkpoints

ENV PYTHONPATH="/app"

ENTRYPOINT ["uv", "run", "-m", "benchmarks"]
CMD ["--fhe-samples", "50"]
