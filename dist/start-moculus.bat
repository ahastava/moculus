@echo off
title MoCoLUS POCUS Trainer
setlocal

set IMAGE=ahastava/moculus:latest
set CONTAINER=moculus

echo.
echo  ============================================
echo   MoCoLUS - Point-of-Care Lung US Simulator
echo  ============================================
echo.

:: Check Docker is installed
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker is not installed.
    echo      Download it from: https://www.docker.com/products/docker-desktop
    echo      Install it, restart your computer, then run this script again.
    echo.
    pause
    exit /b 1
)

:: Check Docker daemon is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo      Waiting for Docker to start (this may take a minute)...
    :wait_docker
    timeout /t 3 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 goto wait_docker
    echo      Docker is ready.
    echo.
)

:: Stop old container if running
docker ps -q -f name=%CONTAINER% >nul 2>&1
for /f %%i in ('docker ps -q -f name^=%CONTAINER%') do (
    echo  Stopping previous session...
    docker stop %CONTAINER% >nul 2>&1
    docker rm %CONTAINER% >nul 2>&1
)

:: Pull latest image
echo  Downloading latest MoCoLUS (first time may take a few minutes)...
docker pull %IMAGE%
if %errorlevel% neq 0 (
    echo.
    echo  [!] Failed to download. Check your internet connection.
    pause
    exit /b 1
)

:: Run container
echo.
echo  Starting MoCoLUS...
docker run -d --name %CONTAINER% -p 8000:8000 %IMAGE% >nul 2>&1

:: Wait for server to be ready
echo  Waiting for server to start...
:wait_server
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/api/scenarios >nul 2>&1
if %errorlevel% neq 0 goto wait_server

:: Open browser
echo.
echo  ============================================
echo   MoCoLUS is running!
echo   Opening browser...
echo.
echo   If it doesn't open, go to:
echo   http://localhost:8000
echo.
echo   To stop: close this window or press Ctrl+C
echo  ============================================
echo.
start http://localhost:8000

:: Keep window open, stop on close
echo  Press any key to stop MoCoLUS...
pause >nul
echo  Stopping MoCoLUS...
docker stop %CONTAINER% >nul 2>&1
docker rm %CONTAINER% >nul 2>&1
echo  Done.
