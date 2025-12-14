#!/bin/bash
set -e

# Configuration
# Configuration
# Parse arguments
FORCE_UPDATE=false
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        FORCE_UPDATE=true
        break
    fi
done

# Configuration
DEFAULT_SERVICE_NAME="abs-recommender"

# Prompt for service name if not provided
if [ -z "$SERVICE_NAME" ]; then
    read -p "Enter service name [$DEFAULT_SERVICE_NAME]: " INPUT_SERVICE_NAME
    SERVICE_NAME=${INPUT_SERVICE_NAME:-$DEFAULT_SERVICE_NAME}
fi
export SERVICE_NAME

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to project root
cd "$PROJECT_ROOT"

# Ensure uv is in PATH (common install location)
export PATH="$HOME/.local/bin:$PATH"

echo "Starting update process for $SERVICE_NAME..."

# 1. Stop the systemd service if it exists
if [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
    echo "Stopping systemd service: $SERVICE_NAME"
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        sudo systemctl stop "$SERVICE_NAME"
    fi
else
    echo "Service $SERVICE_NAME not found (checked /etc/systemd/system/${SERVICE_NAME}.service). Skipping stop."
fi

# 2. Git pull
echo "Pulling latest changes..."
# Capture the original HEAD to compare later
OLD_HEAD=$(git rev-parse HEAD)
git pull
NEW_HEAD=$(git rev-parse HEAD)

# 2.1. Check for self-update
if [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    # Check if this script was updated
    if git diff --name-only "$OLD_HEAD" "$NEW_HEAD" | grep -q "scripts/update.sh"; then
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "Update script updated. Restarting with new version..."
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        exec "$0" "$@"
    fi
fi

# 3. Check diff for python and node requirements
# 3. Check for updates (Git diff or Force)
if [ "$FORCE_UPDATE" = true ] || [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    if [ "$FORCE_UPDATE" = true ]; then
        echo "Force update enabled. Updating all dependencies..."
    else
        echo "Changes detected. Checking for dependency updates..."
    fi

    # ---------------------------
    # Python Checks
    # ---------------------------
    SHOULD_UPDATE_PYTHON=false
    if [ "$FORCE_UPDATE" = true ]; then
        SHOULD_UPDATE_PYTHON=true
    elif git diff --name-only "$OLD_HEAD" "$NEW_HEAD" | grep -E "pyproject.toml|uv.lock"; then
        SHOULD_UPDATE_PYTHON=true
    fi

    if [ "$SHOULD_UPDATE_PYTHON" = true ]; then
        echo "Updating Python dependencies..."
        if command -v uv &> /dev/null; then
            uv sync
        else
            echo "Warning: uv not found in PATH. Attempting to install..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
            uv sync
        fi
    else
        echo "No Python requirement changes."
    fi

    # ---------------------------
    # Frontend Checks
    # ---------------------------
    SHOULD_UPDATE_FRONTEND=false
    SHOULD_INSTALL_NODE=false
    
    if [ "$FORCE_UPDATE" = true ]; then
        SHOULD_UPDATE_FRONTEND=true
        SHOULD_INSTALL_NODE=true
    else
        if git diff --name-only "$OLD_HEAD" "$NEW_HEAD" | grep -E "^frontend/"; then
            SHOULD_UPDATE_FRONTEND=true
        fi
        if git diff --name-only "$OLD_HEAD" "$NEW_HEAD" | grep -E "frontend/package.json|frontend/package-lock.json"; then
            SHOULD_INSTALL_NODE=true
        fi
    fi

    if [ "$SHOULD_UPDATE_FRONTEND" = true ]; then
        echo "Updating Frontend..."
        cd frontend

        # Install if deps changed OR force OR node_modules is missing
        if [ "$SHOULD_INSTALL_NODE" = true ] || [ ! -d "node_modules" ]; then
            echo "Installing dependencies..."
            npm install
        fi

        echo "Building frontend..."
        npm run build
        
        echo "Cleaning up node_modules..."
        rm -rf node_modules
        
        cd ..
    else
        echo "No frontend changes."
    fi
else
    echo "Repo is already up to date."
fi

# 4. Restart systemd if service was created
if [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
    echo "Restarting systemd service: $SERVICE_NAME"
    sudo systemctl start "$SERVICE_NAME"
    sudo systemctl status "$SERVICE_NAME" --no-pager
else
    echo "Service file not found. Skipping restart."
fi

echo "Update process complete!"
