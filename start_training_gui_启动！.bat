@echo off
title Musubi Tuner v0.2.13 Training Interface
echo.
echo ========================================
echo   Musubi Tuner v0.2.13 Training GUI
echo   Authors: suzuki ^& eddy
echo ========================================
echo.

REM Setup Sage4 environment for RTX 5090
echo [INFO] Setting up Sage4 environment for RTX 5090...

REM Set PyTorch lib path
set TORCH_LIB_PATH=%~dp0python_embeded\Lib\site-packages\torch\lib

REM Set CUDA paths
set CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
set CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64

REM Check if paths exist
if exist "%TORCH_LIB_PATH%" (
    echo [OK] PyTorch lib path found
) else (
    echo [WARNING] PyTorch lib path not found
)

if exist "%CUDA_BIN_PATH%" (
    echo [OK] CUDA bin path found
) else (
    echo [WARNING] CUDA bin path not found
)

REM Add paths to PATH environment variable
set PATH=%TORCH_LIB_PATH%;%CUDA_BIN_PATH%;%CUDA_LIB_PATH%;%PATH%

REM Set CUDA environment variables
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

echo [INFO] Environment setup complete
echo [INFO] Available Attention implementations:
echo   - Flash-Attention 2.7.4.post1 (Standard)
echo   - SageAttention 2.2.0 (General)
echo   - Sage4 1.0.0 (RTX 5090 FP4 Optimized)
echo.

echo Starting Gradio interface...
echo Access URL: http://127.0.0.1:7860/
echo.

REM Start training GUI
.\python_embeded\python.exe train_gui_new.py

REM Check for errors
if errorlevel 1 (
    echo.
    echo [ERROR] Program exited with error code: %errorlevel%
    echo Please check the error messages above
    echo.
)

pause
