@echo off
title ArcFace Dataset Sorter - Installation
color 0A

echo ============================================================
echo    ArcFace Dataset Sorter - Installing Dependencies
echo ============================================================
echo.

:: Check if Python is installed
python --version 2>nul
if errorlevel 1 (
    echo [!] Python not found. Attempting to install via winget...
    echo.
    winget install --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [ERROR] Auto-install failed. Please install Python manually from:
        echo         https://www.python.org/downloads/
        echo         Make sure to check "Add Python to PATH"
        pause
        start https://www.python.org/downloads/
        exit /b 1
    )
    echo.
    echo [OK] Python installed. Please RESTART this script to continue.
    pause
    exit /b 0
)

python --version
echo [OK] Python found
echo.

:: Upgrade pip
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip -q

:: Install ONNX Runtime (with GPU auto-detect)
echo [2/5] Installing ONNX Runtime...
pip install onnxruntime -q

:: Install OpenCV
echo [3/5] Installing OpenCV...
pip install opencv-python -q

:: Install InsightFace + Flask
echo [4/5] Installing InsightFace ^& Flask...
pip install insightface flask flask-cors -q

:: Install remaining packages
echo [5/5] Installing remaining packages...
pip install numpy scikit-learn Pillow tqdm scipy -q

echo.
echo ============================================================
echo    Installation Complete!
echo    Run START.bat to launch the application
echo ============================================================
echo.

:: Verify insightface installed
python -c "import insightface; print('[OK] InsightFace', insightface.__version__)" 2>nul || echo [WARN] InsightFace may need manual install

pause
