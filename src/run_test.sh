#!/bin/bash
# Run main.py in the background and ignore SSH disconnection
python main_check_controll.py &

# Capture the exact Process ID (PID) of the Python script and save it
echo $! > main.pid

echo "main.py is now running in the background."
echo "You can safely close SSH. Logs are saving to main.log"