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
        
        # Generate SECRET_KEY
        echo "Generating secure SECRET_KEY..."
        if command -v python3 &>/dev/null; then
            SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
        elif command -v python &>/dev/null; then
            SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
        else
            SECRET_KEY="CHANGE_ME_$(date +%s)"
            echo "Warning: Python not found, using weak SECRET_KEY. Please update it manually."
        fi
        
        # Replace empty SECRET_KEY= with generated key
        # Use a temporary file for compatibility (sed -i differences on Mac/Linux)
        sed "s/SECRET_KEY=/SECRET_KEY=$SECRET_KEY/" .env > .env.tmp && mv .env.tmp .env
        
        echo "✓ .env file created with generated SECRET_KEY"
        echo ""
        echo "IMPORTANT: A new .env file has been created."
        echo "Please edit the .env file with your ABS configuration and run this installation script again."
        exit 1
    else
        echo "Warning: .env.example not found. Please create a .env file and run this installation script again."
        exit 1
    fi
else
    echo "✓ .env file already exists"
    
    # Check if SECRET_KEY exists and is not empty
    if ! grep -q "SECRET_KEY=" .env || grep -q "SECRET_KEY=$" .env; then
        echo "Updating SECRET_KEY in existing .env..."
        if command -v python3 &>/dev/null; then
            SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
        elif command -v python &>/dev/null; then
            SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
        else
            SECRET_KEY="CHANGE_ME_$(date +%s)"
        fi
        
        if grep -q "SECRET_KEY=" .env; then
             sed "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env > .env.tmp && mv .env.tmp .env
        else
             echo "SECRET_KEY=$SECRET_KEY" >> .env
        fi
        echo "✓ SECRET_KEY updated"
    fi
fi

# Security Check: Enforce changing default root password AND secret key check
if [ -f ".env" ]; then
    # Helper function to get value from .env
    get_env_value() {
        grep "^$1=" .env | cut -d'=' -f2-
    }

    CURRENT_PASSWORD=$(get_env_value "ROOT_PASSWORD")
    
    # Check if password is missing (empty) or is the default "admin"
    if [ -z "$CURRENT_PASSWORD" ] || [ "$CURRENT_PASSWORD" = "admin" ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        if [ -z "$CURRENT_PASSWORD" ]; then
            echo "SECURITY WARNING: ROOT_PASSWORD is not set or empty"
        else
            echo "SECURITY WARNING: You are using the default root password (admin)"
        fi
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "You MUST set a secure root password to continue."
        echo ""
        
        while true; do
            read -p "Enter new root password: " NEW_PASSWORD
            
            if [ -z "$NEW_PASSWORD" ]; then
                echo "Password cannot be empty. Please try again."
                continue
            fi
            
            if [ "$NEW_PASSWORD" = "admin" ]; then
                echo "New password cannot be the default 'admin'. Please try again."
                continue
            fi
            
            # If the key exists (even empty), replace it. If not, append it.
            if grep -q "^ROOT_PASSWORD=" .env; then
                sed "s|ROOT_PASSWORD=.*|ROOT_PASSWORD=$NEW_PASSWORD|" .env > .env.tmp && mv .env.tmp .env
            else
                # Ensure we have a newline before appending if the file doesn't end with one
                [ -n "$(tail -c1 .env)" ] && echo "" >> .env
                echo "ROOT_PASSWORD=$NEW_PASSWORD" >> .env
            fi
            
            echo "✓ ROOT_PASSWORD updated successfully."
            break
        done
    fi
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
DEFAULT_WORKERS=2
DEFAULT_THREADS=4
SERVICE_NAME="abs-recommender"

read -p "Enter service name [$SERVICE_NAME]: " INPUT_NAME
SERVICE_NAME=${INPUT_NAME:-$SERVICE_NAME}

read -p "Enter user to run the service [$DEFAULT_USER]: " SERVICE_USER
SERVICE_USER=${SERVICE_USER:-$DEFAULT_USER}

read -p "Enter port to listen on [$DEFAULT_PORT]: " SERVICE_PORT
SERVICE_PORT=${SERVICE_PORT:-$DEFAULT_PORT}

read -p "Enter number of workers [$DEFAULT_WORKERS]: " SERVICE_WORKERS
SERVICE_WORKERS=${SERVICE_WORKERS:-$DEFAULT_WORKERS}

read -p "Enter number of threads [$DEFAULT_THREADS]: " SERVICE_THREADS
SERVICE_THREADS=${SERVICE_THREADS:-$DEFAULT_THREADS}

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
ExecStart=$GUNICORN_PATH --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w $SERVICE_WORKERS --threads $SERVICE_THREADS --bind $DEFAULT_HOST:$SERVICE_PORT web_app.wsgi:application
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