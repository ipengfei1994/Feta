@echo off
rem Feta UI launcher - double click to start, closes when server window is closed
cd /d %~dp0
echo Starting Feta UI (model loading takes ~30s on first run)...
echo Feed:   http://127.0.0.1:8765/
echo Admin:  http://127.0.0.1:8765/dashboard
uv run python ui/server.py
pause
