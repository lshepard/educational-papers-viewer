#!/bin/bash
# Start the Papers Viewer Backend server

# Activate virtual environment if using uv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the server
python main.py
