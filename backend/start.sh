#!/bin/bash
# Start the Papers Viewer Backend server

# Activate virtual environment if using uv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the server with 4 workers for concurrent request handling
# This allows the server to handle multiple requests simultaneously,
# particularly important during long-running podcast generation
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
