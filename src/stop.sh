#!/bin/bash
if [ -f main.pid ]; then
    PID=$(cat main.pid)
    
    # Check if the process is actually still running
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Stopped main.py (PID: $PID)."
    else
        echo "Process $PID was already stopped or crashed. Cleaning up PID file."
    fi
    
    rm main.pid
else
    echo "Error: main.pid not found. Is the script running?"
fi