#!/usr/bin/env bash
# Helper script to monitor the training pipeline.
#
# Usage:
#   bash scripts/monitor_training.sh [status|logs|attach|kill]
#
# Commands:
#   status    Show status of training session
#   logs      Tail the latest training log
#   attach    Attach to the training tmux session
#   kill      Kill the training session
#   help      Show this help message

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="training"
LOG_DIR="${ROOT_DIR}/results/logs"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function show_status() {
  echo "====================================="
  echo "Training Session Status"
  echo "====================================="
  
  if tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    echo -e "${GREEN}✓ Session '${SESSION_NAME}' is ACTIVE${NC}"
    echo ""
    echo "Session details:"
    tmux list-sessions | grep "^${SESSION_NAME}:"
    echo ""
    echo "To attach: tmux attach-session -t ${SESSION_NAME}"
  else
    echo -e "${RED}✗ Session '${SESSION_NAME}' is NOT running${NC}"
    echo ""
    echo "To start: bash scripts/run_pipeline_tmux.sh"
  fi
  echo ""
  
  # Show latest checkpoint
  if [[ -d "${ROOT_DIR}/checkpoints" ]]; then
    LATEST_CHECKPOINT=$(ls -1dt "${ROOT_DIR}/checkpoints"/*/ 2>/dev/null | head -1)
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      echo "Latest checkpoint:"
      echo "  ${LATEST_CHECKPOINT}"
      STEP_FILE="${LATEST_CHECKPOINT}/trainer_state.json"
      if [[ -f "${STEP_FILE}" ]]; then
        GLOBAL_STEP=$(grep -o '"global_step": [0-9]*' "${STEP_FILE}" | head -1 | grep -o '[0-9]*')
        echo "  Global steps completed: ${GLOBAL_STEP}"
      fi
    fi
  fi
  echo ""
}

function show_logs() {
  if [[ ! -d "${LOG_DIR}" ]]; then
    echo -e "${YELLOW}No logs directory yet. Training may not have started.${NC}"
    return 1
  fi
  
  LATEST_LOG=$(ls -1t "${LOG_DIR}"/training_*.log 2>/dev/null | head -1)
  if [[ -z "${LATEST_LOG}" ]]; then
    echo -e "${YELLOW}No log files found yet.${NC}"
    return 1
  fi
  
  echo "====================================="
  echo "Latest Log: ${LATEST_LOG}"
  echo "====================================="
  tail -100 "${LATEST_LOG}"
}

function attach_session() {
  if ! tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    echo -e "${RED}Error: Session '${SESSION_NAME}' is not running.${NC}"
    echo "Start with: bash scripts/run_pipeline_tmux.sh"
    exit 1
  fi
  
  echo "Attaching to session '${SESSION_NAME}'..."
  echo "(Press Ctrl+B then D to detach)"
  tmux attach-session -t "${SESSION_NAME}"
}

function kill_session() {
  if ! tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    echo -e "${YELLOW}Session '${SESSION_NAME}' is not running.${NC}"
    return 0
  fi
  
  read -p "Kill training session '${SESSION_NAME}'? (y/n) " -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux kill-session -t "${SESSION_NAME}"
    echo -e "${GREEN}Session killed.${NC}"
  else
    echo "Cancelled."
  fi
}

function show_help() {
  cat << EOF
Monitor and manage the training pipeline.

Usage:
  bash scripts/monitor_training.sh [command]

Commands:
  status              Show training session status
  logs                Show latest training log (tail -100)
  attach              Attach to the training tmux session
  kill                Kill the training session
  help                Show this help message

Examples:
  # Check if training is running
  bash scripts/monitor_training.sh status
  
  # Watch the logs
  bash scripts/monitor_training.sh logs
  
  # Reconnect to training
  bash scripts/monitor_training.sh attach
EOF
}

# Main
COMMAND="${1:-status}"

case "${COMMAND}" in
  status)
    show_status
    ;;
  logs)
    show_logs
    ;;
  attach)
    attach_session
    ;;
  kill)
    kill_session
    ;;
  help)
    show_help
    ;;
  *)
    echo "Unknown command: ${COMMAND}"
    echo "Run 'bash scripts/monitor_training.sh help' for usage."
    exit 1
    ;;
esac
