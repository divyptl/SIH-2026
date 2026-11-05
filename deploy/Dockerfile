# syntax=docker/dockerfile:1
#
# SatQuery AI backend, CPU-only. Builds for x86-64 and ARM64 (Raspberry Pi 5).
# The build context is the repository root:
#
#   docker build -f deploy/Dockerfile -t satquery-api .
#
# The fine-tuned checkpoints are not in the image: start.sh downloads them from
# the WEIGHTS_REPO model repo on the Hub when the container starts.

FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
# IndicTransToolkit ships as an sdist with a Cython extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml backend/README.md backend/
COPY ml/pyproject.toml ml/README.md ml/

# The lockfile pins CUDA 13 torch for Linux. Swap it for the CPU build of the same
# version and drop the CUDA runtime packages: there is no GPU, and this keeps
# several GB out of the image.
RUN uv export --package backend --frozen --no-hashes --no-emit-workspace --no-header -o requirements.txt \
    && sed -i -E -e 's/\+cu130/+cpu/' -e '/^(nvidia-|cuda-|triton)/d' requirements.txt

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install -r requirements.txt \
    # Commented out of backend/pyproject.toml, but translation needs it.
    && VIRTUAL_ENV=/opt/venv uv pip install "indictranstoolkit==1.1.1" -c requirements.txt


FROM python:3.14-slim

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app/checkpoints /home/user/.cache/huggingface \
    && chown -R user:user /home/user
USER user

ENV HOME=/home/user \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    OMP_NUM_THREADS=2 \
    # Checkpoints land in ./checkpoints (see fetch_weights.py); these mirror backend/.env.
    GROUNDING_CHECKPOINT=checkpoints/grounding/v3/last.pt \
    GROUNDING_RERANKERS=checkpoints/grounding/combined/reranker.pt,checkpoints/grounding/combined/reranker_seed1.pt,checkpoints/grounding/combined/reranker_seed2.pt,checkpoints/grounding/combined/reranker_seed3.pt,checkpoints/grounding/combined/reranker_seed4.pt \
    CHANGE_VQA_CHECKPOINT=checkpoints/change_detection_v2/best.pt \
    FUSION_CHECKPOINT=checkpoints/fusion_best.pt \
    SPECIALIST_DEVICE=cpu \
    TRANSLATION_DEVICE=cpu \
    SPECIALIST_PRELOAD=true \
    TRANSLATION_PRELOAD=true

COPY --from=build --chown=user /opt/venv /opt/venv

WORKDIR /home/user/app

# GroundingDINO's base weights are public; bake them in so startup skips the download.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny')"

COPY --chown=user backend backend
COPY --chown=user ml ml
COPY --chown=user deploy/start.sh deploy/fetch_weights.py ./

EXPOSE 7860
CMD ["sh", "./start.sh"]
