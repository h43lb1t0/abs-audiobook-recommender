#!/bin/bash

# Install script for abs-audiobook-recommender
# This script installs all dependencies for both the backend and frontend

set -e  # Exit on any error

echo "====================================="
echo "ABS Audiobook Recommender - Installer"
echo "====================================="
echo ""


# Function to install uv
install_uv() {
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell config to get uv in PATH
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        echo "✓ uv installed successfully"
    else
        echo "Error: Failed to install uv"
        exit 1
    fi
}


get_python_version() {
    if [ ! -f "pyproject.toml" ]; then
        echo "Error: pyproject.toml not found" >&2
        return 1
    fi
    
    # Extract version (removes >=, <, etc and quotes)
    local version
    version=$(grep "requires-python" pyproject.toml | cut -d'"' -f2 | sed 's/[^0-9.]//g')
    
    if [ -z "$version" ]; then
        echo "Error: Could not find requires-python in pyproject.toml" >&2
        return 1
    fi
    
    echo "$version"
}

install_python() {
    echo "Reading Python version from pyproject.toml..."
    
    PYTHON_VERSION=$(get_python_version)
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    echo "Found Python version requirement: $PYTHON_VERSION"
    echo "Installing Python $PYTHON_VERSION using uv..."
    uv python install "$PYTHON_VERSION"
    
    if [ $? -eq 0 ]; then
        echo "✓ Python $PYTHON_VERSION installed successfully"
    else
        echo "Error: Failed to install Python"
        exit 1
    fi
}

# Function to install Node.js via nvm
install_node() {
    echo "Installing Node.js via nvm..."
    
    # Install nvm if not present
    if [ ! -d "$HOME/.nvm" ]; then
        echo "Installing nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    fi
    
    # Source nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install latest LTS Node.js
    nvm install --lts
    nvm use --lts
    
    if command -v node &> /dev/null; then
        echo "✓ Node.js installed successfully ($(node --version))"
    else
        echo "Error: Failed to install Node.js"
        exit 1
    fi
}


echo "====================================="
echo "Checking for dependencies..."
echo "and install if needed..."
echo "The following dependencies will be installed:"
echo "- uv"
echo "- Python (from pyproject.toml)"
echo "- Node.js"
echo "- npm"
echo "====================================="
echo ""

echo "Press enter to continue or any key to exit..."
read -r

if [ -z "$REPLY" ]; then
    echo "Continuing..."
else
    echo "Exiting..."
    exit 0
fi

# Check and install uv if needed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    install_uv
else
    echo "✓ uv found"
fi

# Install Python

# Check if we need to install Python
PYTHON_VERSION=$(get_python_version)
NEED_INSTALL=true

if [ -n "$PYTHON_VERSION" ]; then
    # Check python3
    if command -v python3 &> /dev/null; then
        if python3 -c "import sys; v='$PYTHON_VERSION'.split('.'); req=tuple(map(int, v)); cur=sys.version_info[:len(req)]; sys.exit(0 if cur >= req else 1)" 2>/dev/null; then
            echo "✓ Found compatible system Python: $(python3 --version)"
            NEED_INSTALL=false
        fi
    fi
    
    # If python3 didn't satisfy, try python
    if [ "$NEED_INSTALL" = true ] && command -v python &> /dev/null; then
         if python -c "import sys; v='$PYTHON_VERSION'.split('.'); req=tuple(map(int, v)); cur=sys.version_info[:len(req)]; sys.exit(0 if cur >= req else 1)" 2>/dev/null; then
            echo "✓ Found compatible system Python: $(python --version)"
            NEED_INSTALL=false
        fi
    fi
fi

if [ "$NEED_INSTALL" = true ]; then
    install_python
fi

# Check and install Node.js if needed
echo "Checking for Node.js..."
if ! command -v node &> /dev/null; then
    echo "Node.js not found. Installing..."
    install_node
else
    echo "✓ Node.js found ($(node --version))"
fi

# npm comes with Node.js, just verify
echo "Checking for npm..."
if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. This should have been installed with Node.js."
    exit 1
fi
echo "✓ npm found ($(npm --version))"

echo ""

# Install Python dependencies
echo "====================================="
echo "Installing Python dependencies..."
echo "====================================="
uv sync
echo "✓ Python dependencies installed"

echo ""

# Install Frontend dependencies
echo "====================================="
echo "Installing Frontend dependencies..."
echo "====================================="
cd frontend
npm install
echo "✓ Frontend dependencies installed"

cd ..
echo ""
echo "====================================="
echo "Installation complete!"
echo "====================================="
echo ""
