#!/bin/sh
# Pull the latest API image and restart the container if it changed. Run by cron
# every few minutes (see deploy/README.md); safe to run by hand at any time.
set -e
cd "$(dirname "$0")"

docker compose pull --quiet api
# Recreates the container only when the pulled image differs from the running one.
docker compose up -d
docker image prune -f >/dev/null
