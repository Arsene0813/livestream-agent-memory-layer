#!/usr/bin/env bash
set -e
cd ~/agent/openclawd-infra
docker compose down
echo "Stopped (data volumes kept)."
