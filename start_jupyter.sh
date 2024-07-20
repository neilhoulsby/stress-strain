#!/bin/bash

# Name of the virtual environment
ENV_NAME="ml_env"

# Path to the virtual environment
ENV_PATH="$HOME/$ENV_NAME"

# Activate the virtual environment
source "$ENV_PATH/bin/activate"

# Start Jupyter Lab
nohup jupyter lab --no-browser --ip=0.0.0.0 --port=8888 > "$HOME/jupyter.log" 2>&1 &

# Get the process ID of the Jupyter Lab instance
JUPYTER_PID=$!

# Wait a bit for Jupyter to start
sleep 5

# Check if Jupyter is running
if ps -p $JUPYTER_PID > /dev/null
then
    echo "Jupyter Lab started successfully. PID: $JUPYTER_PID"
    echo "Check $HOME/jupyter.log for the URL and token."
    echo "To stop Jupyter, run: kill $JUPYTER_PID"
else
    echo "Failed to start Jupyter Lab. Check $HOME/jupyter.log for details."
fi

# Deactivate the virtual environment
deactivate
