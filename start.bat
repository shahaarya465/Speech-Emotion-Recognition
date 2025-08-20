@echo off
TITLE Speech Emotion Recognition Launcher

:: --- Dependency Check ---
ECHO Checking for required software...

:: Check for Python 3.12
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    ECHO.
    ECHO #################################################################
    ECHO # ERROR: Python 3.12 is not found. Please install it.           #
    ECHO #################################################################
    ECHO.
    pause
    exit /b
)

:: Check for Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    ECHO.
    ECHO #################################################################
    ECHO # ERROR: Node.js is not found. Please install it.               #
    ECHO #################################################################
    ECHO.
    pause
    exit /b
)

ECHO Dependencies found.

:: --- One-Time Setup ---
IF NOT EXIST ".venv" (
    ECHO.
    ECHO --- Performing one-time setup ---
    ECHO Creating Python virtual environment...
    py -3.12 -m venv .venv

    ECHO Activating environment and installing Python packages...
    call .venv\Scripts\activate.bat
    pip install -r backend\requirements.txt

    ECHO Installing Node.js packages...
    pushd frontend
    call npm install
    popd
    ECHO --- Setup complete! ---
    ECHO.
)


:: --- Application Startup ---
ECHO Starting application servers...

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Start the Python backend in a new window
ECHO Starting Python backend server...
start "SER-Backend" python backend\app.py

:: Start the React frontend in a new window
ECHO Starting React frontend server...
pushd frontend
start "SER-Frontend" npm run dev
popd

ECHO Waiting for servers to initialize...
timeout /t 8 >nul

ECHO Opening application in your browser at http://localhost:5173
start http://localhost:5173

ECHO.
ECHO #################################################################
ECHO # Your application is now running!                              #
ECHO # To stop the application, close all three command windows.     #
ECHO #################################################################
ECHO.
pause
