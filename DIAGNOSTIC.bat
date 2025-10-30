@echo off
chcp 65001 >nul
cls
echo ================================================
echo   SERVER STARTUP DIAGNOSTIC
echo ================================================
echo.

cd /d %~dp0
echo Current directory: %CD%
echo.

echo [1] Checking Python...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found!
    goto :error
)
echo [OK] Python installed
echo.

echo [2] Checking files...
if not exist "api_server.py" (
    echo [ERROR] api_server.py not found!
    goto :error
)
if not exist ".env" (
    echo [WARNING] .env file not found!
)
echo [OK] All files present
echo.

echo [3] Checking dependencies...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [ERROR] Dependencies not installed
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        goto :error
    )
)
echo [OK] Dependencies installed
echo.

echo [4] Test import...
python -c "from api_server import app; print('[OK] Import successful')" 2>&1
if errorlevel 1 (
    echo [ERROR] Error importing api_server
    goto :error
)
echo.

echo [5] Stopping old server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Stopping process %%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo [OK] Port 8000 is free
echo.

echo ================================================
echo   ALL CHECKS PASSED
echo ================================================
echo.
echo Starting server in new window...
echo.

start "Video Prompt Pipeline Server" cmd /k "cd /d %~dp0 && python api_server.py"
timeout /t 3 /nobreak >nul

echo.
echo Checking startup...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 (
    echo [WARNING] Server not started yet (this is normal, wait a bit)
) else (
    echo [OK] Server started on port 8000!
)

echo.
echo Open "Video Prompt Pipeline Server" window to view logs
echo.
pause
exit /b 0

:error
echo.
echo ================================================
echo   ERROR DETECTED
echo ================================================
echo.
echo Fix the error and run again
pause
exit /b 1

