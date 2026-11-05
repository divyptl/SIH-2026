#!/bin/sh
# Container entrypoint: fetch the checkpoints, then serve the API on the Space port.
set -e

python fetch_weights.py

cd backend
# One worker: every worker would load its own copy of the models.
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
