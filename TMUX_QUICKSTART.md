# Training with tmux - Quick Start Guide

## Why tmux?

`tmux` creates a **persistent session** that keeps running even if:
- Your SSH connection drops
- Your laptop goes to sleep/hibernates
- Your internet disconnects
- You close your terminal

## Quick Start

### 1. Start Training (in background)

```bash
bash scripts/run_pipeline_tmux.sh
```

This starts training in a detached tmux session named `training`.

**Output:**
```
Training started in tmux session: training

To monitor training:
  tmux attach-session -t training

To view recent logs:
  tail -f results/logs/training_*.log
```

### 2. Monitor Status

```bash
bash scripts/monitor_training.sh status
```

Shows:
- ✓ Whether training is active
- Latest checkpoint progress
- How to reconnect

### 3. View Logs

```bash
bash scripts/monitor_training.sh logs
```

Shows the last 100 lines of the current training log. Or use:

```bash
tail -f results/logs/training_*.log
```

to stream logs in real-time.

### 4. Reconnect After Disconnection

```bash
tmux attach-session -t training
```

This re-attaches your terminal to the running session.

**To leave without stopping training:**
- Press `Ctrl+B`, then `D` (detach)

### 5. Stop Training (if needed)

```bash
bash scripts/monitor_training.sh kill
```

Or directly:
```bash
tmux kill-session -t training
```

---

## Advanced Usage

### Custom Session Name

```bash
bash scripts/run_pipeline_tmux.sh my_experiment
tmux attach-session -t my_experiment
```

### List All Sessions

```bash
tmux list-sessions
```

Example output:
```
my_experiment: 1 windows (created Sat Apr 5 14:30:20 2026)
training: 1 windows (created Sat Apr 5 14:20:35 2026)
```

### View Session Details

```bash
tmux capture-pane -t training -p | head -50
```

Shows the last 50 lines of the session (equivalent to scrolling up in the terminal).

---

## Log Files

All output is automatically saved to:
```
results/logs/training_YYYYMMDD_HHMMSS.log
```

Example:
```
results/logs/training_20260405_143500.log
```

View all logs:
```bash
ls -lh results/logs/
```

---

## Troubleshooting

### tmux not installed?

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y tmux
```

**macOS:**
```bash
brew install tmux
```

**Windows (WSL2):**
```bash
sudo apt-get update && sudo apt-get install -y tmux
```

### Can't reconnect to session?

```bash
# List all sessions
tmux list-sessions

# Try to attach
tmux attach-session -t training

# If that fails, the session might be dead. Restart:
bash scripts/run_pipeline_tmux.sh
```

### Port conflicts or old zombie processes?

```bash
# Kill all training processes (careful!)
pkill -f "train_attacks.py"

# Or kill just the tmux session
tmux kill-session -t training
```

---

## Monitoring Commands Summary

| Command | Purpose |
|---------|---------|
| `bash scripts/run_pipeline_tmux.sh` | Start training in tmux |
| `bash scripts/monitor_training.sh status` | Check if training is running |
| `bash scripts/monitor_training.sh logs` | View latest log tail |
| `bash scripts/monitor_training.sh attach` | Reconnect to session |
| `tmux attach-session -t training` | Manually attach to session |
| `tmux list-sessions` | List all active sessions |
| `tail -f results/logs/training_*.log` | Stream logs in real-time |
| `Ctrl+B D` | Detach from session (inside tmux) |

---

## Example Workflow

```bash
# 1. Start training
bash scripts/run_pipeline_tmux.sh

# 2. Check status (while training runs)
bash scripts/monitor_training.sh status

# 3. Laptop goes to sleep... no problem!
# (Connection drops but training continues in tmux)

# 4. Reconnect later
tmux attach-session -t training

# 5. View logs
bash scripts/monitor_training.sh logs

# 6. Detach and let training continue
# (Press Ctrl+B then D)

# 7. When done, kill session
bash scripts/monitor_training.sh kill
```
