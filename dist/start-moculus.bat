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
if errorlevel 1 (
    echo  [!] Docker is not installed.
    echo      Download it from: https://www.docker.com/products/docker-desktop
    echo      Install it, restart your computer, then run this script again.
    echo.
    pause
    exit /b 1
)

:: Check Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo  [!] Docker is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo      Waiting for Docker to start. This may take a minute...
    call :wait_for_docker
    echo      Docker is ready.
    echo.
)

:: Stop old container if running (errors suppressed if none exists)
docker stop %CONTAINER% >nul 2>&1
docker rm %CONTAINER% >nul 2>&1

:: Pull latest image
echo  Downloading latest MoCoLUS ^(first time may take a few minutes^)...
docker pull %IMAGE%
if errorlevel 1 (
    echo.
    echo  [!] Failed to download. Check your internet connection.
    pause
    exit /b 1
)

:: Clean up old dangling images to save disk space
docker image prune -f >nul 2>&1

:: Run container
echo.
echo  Starting MoCoLUS...
docker run -d --name %CONTAINER% -p 8000:8000 %IMAGE% >nul 2>&1

:: Wait for server to be ready
echo  Waiting for server to start...
call :wait_for_server

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
exit /b 0

:: ----- Subroutines -----
:: Labels must live OUTSIDE any if/for blocks to work reliably.

:wait_for_docker
timeout /t 3 /nobreak >nul
docker info >nul 2>&1
if errorlevel 1 goto :wait_for_docker
goto :eof

:wait_for_server
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/api/scenarios >nul 2>&1
if errorlevel 1 goto :wait_for_server
goto :eof
