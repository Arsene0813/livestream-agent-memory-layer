#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[1/3] Starting services..."
docker compose up -d

echo "[2/3] Container status:"
docker compose ps

echo "[3/3] Health checks:"
curl -s http://127.0.0.1:8000/health | sed 's/^/API: /'
echo
curl -s http://127.0.0.1:6333/healthz && echo "Qdrant: healthz passed"
