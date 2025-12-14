#!/bin/bash
set -e

# Get the directory of this script to safely find other scripts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "----------------------------------------------------------------"
echo "Uninstall Script"
echo "----------------------------------------------------------------"

# Default service name
DEFAULT_SERVICE_NAME="abs-recommender"

read -p "Enter service name to uninstall [$DEFAULT_SERVICE_NAME]: " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-$DEFAULT_SERVICE_NAME}

if command -v systemctl &> /dev/null; then
    # Check if service unit exists before trying to stop/disable
    if sudo systemctl list-unit-files "$SERVICE_NAME.service" &> /dev/null; then
        echo "Found service: $SERVICE_NAME"
        
        if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "Stopping service $SERVICE_NAME..."
            ls_output=$(sudo systemctl stop "$SERVICE_NAME" 2>&1) || echo "Warning: Failed to stop service: $ls_output"
        fi

        if sudo systemctl is-enabled --quiet "$SERVICE_NAME"; then
            echo "Disabling service $SERVICE_NAME..."
            ls_output=$(sudo systemctl disable "$SERVICE_NAME" 2>&1) || echo "Warning: Failed to disable service: $ls_output"
        fi
    else
        echo "Service $SERVICE_NAME not found in systemd."
    fi
    
    SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
    if [ -f "$SERVICE_FILE" ]; then
        echo "Removing service file $SERVICE_FILE..."
        sudo rm "$SERVICE_FILE"
        echo "Reloading systemd daemon..."
        sudo systemctl daemon-reload
        echo "✓ Service removed"
    else
        echo "Service file $SERVICE_FILE not found."
    fi
else
    echo "Warning: systemctl not found. skipping service removal."
fi

# Remove .venv
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    read -p "Remove virtual environment at $VENV_PATH? (y/N) " REMOVE_VENV
    if [[ "$REMOVE_VENV" =~ ^[Yy]$ ]]; then
        echo "Removing virtual environment..."
        rm -rf "$VENV_PATH"
        echo "✓ .venv removed"
    else
        echo "Keeping virtual environment."
    fi
else
    echo "Virtual environment not found at $VENV_PATH"
fi

echo ""
echo "Uninstallation complete!"
