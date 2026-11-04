#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/shaoshitong/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU="$1"
[[ "$GPU" == 6 || "$GPU" == 7 ]]
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/twohead-lr10-preview50.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/twohead-lr10-preview50.exit"; printf "preview50_exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)"; exit "$rc"' EXIT
if [[ -f "$ROOT/control/twohead-lr10-preview50.pid" ]]; then
 previous="$(cat "$ROOT/control/twohead-lr10-preview50.pid")"
 if kill -0 "$previous" 2>/dev/null; then echo 'Preview already active'; exit 4; fi
fi
printf '%s\n' "$$" >"$ROOT/control/twohead-lr10-preview50.pid"
source /data/shaoshitong/RDM-twohead-lr10-v1/ops/h20-data-env.sh
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/data/shaoshitong/RDM-twohead-lr10-v1/z-image:/data/shaoshitong/RDM-twohead-lr10-v1
cd /data/shaoshitong/RDM-twohead-lr10-v1
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-preview-twohead50.py" --gpu "$GPU" --cap-gib 32
