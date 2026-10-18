@echo off
title ArcFace Dataset Sorter
color 0B

echo ============================================================
echo    ArcFace Dataset Sorter - Starting Server...
echo ============================================================
echo.

:: Check Python
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found! Please run INSTALL.bat first
    pause
    exit /b 1
)

:: Create required folders
if not exist "Sorted" mkdir "Sorted"
if not exist "Waiting" mkdir "Waiting"
if not exist ".face_db" mkdir ".face_db"
if not exist ".thumbnails" mkdir ".thumbnails"

echo [OK] Folders ready
echo [OK] Starting Flask server...
echo.
echo ============================================================
echo    Open browser: http://localhost:5000
echo    Press Ctrl+C to stop
echo ============================================================
echo.

:: Start browser after short delay
start /min cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Start the Flask server
python app.py

echo.
echo Server stopped.
pause
