#!/bin/sh
echo "Creating Roles and Admin User on the server..."
python /app/roles.py

echo "Starting FastAPI server..."
python -m uvicorn pytupli.server.main:app --host 0.0.0.0 --port $PORT
