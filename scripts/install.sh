#!/bin/bash
set -e

# Get the directory of this script to safely find other scripts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to project root to ensure consistent execution context
cd "$PROJECT_ROOT"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ .env file created"
    else
        echo "Warning: .env.example not found. Skipping .env creation."
    fi
else
    echo "✓ .env file already exists"
fi

# 1. Install dependencies
echo "Calling install_dependencies.sh..."
source "$SCRIPT_DIR/install_dependencies.sh"

# 2. Build frontend
echo "Start building the frontend..."
cd frontend
npm run build
echo "✓ Frontend built"

# 3. Cleanup
echo "Remove node_modules..."
rm -rf node_modules
echo "✓ node_modules removed"

cd "$PROJECT_ROOT"

# 4. Systemd Service
echo ""
echo "----------------------------------------------------------------"
echo "Systemd Service Creation"
echo "----------------------------------------------------------------"
echo "To run this application as a background service, we can create a systemd service file."

read -p "Do you want to create a systemd service? (y/N) " CREATE_SERVICE
if [[ ! "$CREATE_SERVICE" =~ ^[Yy]$ ]]; then
    echo "Skipping systemd service creation."
    echo "Installation complete!"
    exit 0
fi

# Default values
DEFAULT_USER=$(whoami)
DEFAULT_PORT=5000
DEFAULT_HOST="0.0.0.0"
SERVICE_NAME="abs-recommender"

read -p "Enter service name [$SERVICE_NAME]: " INPUT_NAME
SERVICE_NAME=${INPUT_NAME:-$SERVICE_NAME}

read -p "Enter user to run the service [$DEFAULT_USER]: " SERVICE_USER
SERVICE_USER=${SERVICE_USER:-$DEFAULT_USER}

read -p "Enter port to listen on [$DEFAULT_PORT]: " SERVICE_PORT
SERVICE_PORT=${SERVICE_PORT:-$DEFAULT_PORT}

VENV_PATH="$PROJECT_ROOT/.venv"
GUNICORN_PATH="$VENV_PATH/bin/gunicorn"

if [ ! -f "$GUNICORN_PATH" ]; then
    echo "Error: Gunicorn executable not found at $GUNICORN_PATH"
    echo "Make sure dependencies are installed correctly."
    exit 1
fi

SERVICE_FILE_CONTENT="[Unit]
Description=ABS Audiobook Recommender
After=network.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
Environment=\"PATH=$VENV_PATH/bin:/usr/local/bin:/usr/bin:/bin\"
ExecStart=$GUNICORN_PATH --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 --bind $DEFAULT_HOST:$SERVICE_PORT web_app.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target"

echo ""
echo "Proposed /etc/systemd/system/$SERVICE_NAME.service:"
echo "----------------------------------------------------------------"
echo "$SERVICE_FILE_CONTENT"
echo "----------------------------------------------------------------"

read -p "Install this service file? (requires sudo) (y/N) " INSTALL_SERVICE
if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    TEMP_FILE=$(mktemp)
    echo "$SERVICE_FILE_CONTENT" > "$TEMP_FILE"
    
    echo "Installing service file..."
    if sudo cp "$TEMP_FILE" "/etc/systemd/system/$SERVICE_NAME.service"; then
        rm "$TEMP_FILE"
        echo "✓ Service file created at /etc/systemd/system/$SERVICE_NAME.service"
        
        if command -v systemctl &> /dev/null; then
            echo "Reloading systemd daemon..."
            sudo systemctl daemon-reload
            
            read -p "Enable and start the service now? (y/N) " START_SERVICE
            if [[ "$START_SERVICE" =~ ^[Yy]$ ]]; then
                sudo systemctl enable "$SERVICE_NAME"
                sudo systemctl start "$SERVICE_NAME"
                echo "✓ Service enabled and started"
                sudo systemctl status "$SERVICE_NAME" --no-pager
            fi
        else
            echo "Warning: systemctl not found. Please enable the service manually."
        fi
    else
        echo "Error: Failed to copy service file (permission denied?)"
        rm "$TEMP_FILE"
        exit 1
    fi
else
    echo "Skipping service installation."
fi

echo ""
echo "Installation complete!"