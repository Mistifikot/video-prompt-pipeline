@echo off
chcp 65001 >nul
cls
echo ============================================
echo   VIDEO PROMPT PIPELINE
echo   Full System Startup
echo ============================================
echo.

cd /d %~dp0

echo Step 1/4: Checking .env file...
python fix_env.py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to check .env
    echo Run: python fix_env.py
    pause
    exit /b 1
)

echo Step 2/4: Checking dependencies...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Step 3/4: Stopping old server if running...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Step 4/4: Starting server...
cd /d %~dp0
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    pause
    exit /b 1
)
echo Checking api_server.py...
if not exist "api_server.py" (
    echo ERROR: api_server.py not found!
    echo Current directory: %CD%
    pause
    exit /b 1
)
echo Starting server in new window...
start "Video Prompt Pipeline Server" cmd /k "cd /d %~dp0 && python api_server.py"

echo.
echo Waiting for server startup...
timeout /t 5 /nobreak >nul

echo.
echo Checking server...
timeout /t 2 /nobreak >nul
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 (
    echo WARNING: Server not started yet (wait a bit)
) else (
    echo OK: Server started on port 8000!
)

echo.
echo ============================================
echo   SERVER STARTED
echo ============================================
echo.
echo Available interfaces:
echo.
echo   1. Video/Image Analysis:
echo      http://localhost:8000/ui
echo      Or open: web_interface.html
echo.
echo   2. Veo 3.1 - Video Generation:
echo      http://localhost:8000/veo31
echo      Or open: veo31_interface.html
echo.
echo Opening both interfaces...
timeout /t 2 /nobreak >nul
start "" "web_interface.html"
timeout /t 1 /nobreak >nul
start "" "veo31_interface.html"

echo.
echo ============================================
echo   DONE!
echo ============================================
echo.
echo Both interfaces opened in browser
echo Server running in separate window
echo.
echo To stop server, close "Video Prompt Pipeline Server" window
echo.
pause

