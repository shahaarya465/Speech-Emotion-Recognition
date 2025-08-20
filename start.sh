#!/bin/bash
echo "Starting the Speech Emotion Recognition App..."

# --- Dependency Check ---
# This function checks if a command exists.
command_exists () {
    type "$1" &> /dev/null ;
}

# Check for Python 3.12
if ! command_exists python3.12 ; then
    echo ""
    echo "#################################################################"
    echo "# ERROR: python3.12 could not be found."
    echo "# Please install Python 3.12.3 and make sure it's in your PATH. #"
    echo "#################################################################"
    echo ""
    exit 1
fi

# Check for Node.js
if ! command_exists node ; then
    echo ""
    echo "#################################################################"
    echo "# ERROR: Node.js could not be found."
    echo "# Please install Node.js (LTS version) from nodejs.org.         #"
    echo "#################################################################"
    echo ""
    exit 1
fi
echo "Dependencies found."

# --- One-Time Setup ---
# This block only runs if the .venv folder does not exist.
if [ ! -d ".venv" ]; then
    echo ""
    echo "--- Performing one-time setup ---"
    echo "Creating Python virtual environment using Python 3.12..."
    python3.12 -m venv .venv

    echo "Activating environment and installing Python packages..."
    source .venv/bin/activate
    pip install -r backend/requirements.txt
    deactivate

    echo "Installing Node.js packages..."
    (cd frontend && npm install)
    echo "--- Setup complete! ---"
    echo ""
fi


# --- Application Startup ---
echo "Starting application servers..."

# Activate the virtual environment for this session
source .venv/bin/activate

# Start the Python backend in the background
echo "Starting Python backend server (in background)..."
python backend/app.py &
BACKEND_PID=$!

# Start the React frontend
echo "Starting React frontend server..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo "Waiting for servers to initialize..."
sleep 8

echo "Opening application in your browser at http://localhost:5173"
# Use 'open' on macOS, 'xdg-open' on Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
  open http://localhost:5173
else
  xdg-open http://localhost:5173
fi

echo ""
echo "#################################################################"
echo "# Your application is now running!"
echo "# - Backend is on port 5000 (PID: $BACKEND_PID)"
echo "# - Frontend is on port 5173 (PID: $FRONTEND_PID)"
echo "# Press Ctrl+C in this terminal to stop both servers."
echo "#################################################################"
echo ""

# This function will run when the script is exited (e.g., with Ctrl+C)
cleanup() {
    echo -e "\nShutting down background servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}
trap cleanup EXIT

# Wait for the user to stop the script
wait
