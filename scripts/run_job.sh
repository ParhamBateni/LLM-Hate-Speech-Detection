#!/usr/bin/env bash
# Start main.py in a detached tmux session; stdout/stderr go to logs/<run_time>/main.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="$(date +'%m_%d_%H:%M:%S')"
LOG_DIR="${ROOT}/logs/"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
SESSION_NAME="hsc-${RUN_ID//:/-}"

mkdir -p "$LOG_DIR"

if [[ -x "/venv/main/bin/python" ]]; then
  PYTHON="/venv/main/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n main \
  "cd '$ROOT' && exec $PYTHON src/main.py 2>&1 | tee '$LOG_FILE'"

echo "Started tmux session: $SESSION_NAME"
echo "Log directory: $LOG_DIR"
echo "Log file: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f '$LOG_FILE'"
