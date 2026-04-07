#!/usr/bin/env bash
# Wrapper to run the training pipeline in a persistent tmux session.
# Survives SSH disconnections and laptop sleep/shutdown.
#
# Usage:
#   bash scripts/run_pipeline_tmux.sh [tmux_session_name]
#
# Default session name: "training"
#
# After starting, reconnect to the session with:
#   tmux attach-session -t training
#
# To detach without stopping the session (press Ctrl+B then D):
#   [Ctrl+B] [D]
#
# To kill the session (stops training):
#   tmux kill-session -t training
#
# To list all tmux sessions:
#   tmux list-sessions

set -e

# Get project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_SCRIPT="${ROOT_DIR}/scripts/run_pipeline.sh"

# Allow custom session name via argument, default to "training"
SESSION_NAME="${1:-training}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
  echo "ERROR: tmux is not installed. Install with:"
  echo "  Ubuntu/Debian: sudo apt-get install tmux"
  echo "  macOS: brew install tmux"
  echo "  Windows (WSL): sudo apt-get install tmux"
  exit 1
fi

# Kill existing session with same name if it exists
if tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
  echo "Killing existing tmux session '${SESSION_NAME}'..."
  tmux kill-session -t "${SESSION_NAME}"
fi

# Create new detached session and run pipeline
echo "Starting training in tmux session '${SESSION_NAME}'..."
tmux new-session -d -s "${SESSION_NAME}" \
  -x 200 -y 50 \
  -c "${ROOT_DIR}" \
  "bash ${PIPELINE_SCRIPT}"

echo ""
echo "====================================="
echo "Training started in tmux session: ${SESSION_NAME}"
echo "====================================="
echo ""
echo "To monitor training:"
echo "  tmux attach-session -t ${SESSION_NAME}"
echo ""
echo "To detach without stopping:"
echo "  Press Ctrl+B, then D"
echo ""
echo "To view recent logs (if training already started):"
echo "  tail -f results/logs/training_*.log"
echo ""
echo "To list all tmux sessions:"
echo "  tmux list-sessions"
echo ""
echo "To kill the training session:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo "====================================="
