#!/bin/bash

# Use python or python3 depending on availability
python_exe=python
if ! command -v $python_exe &> /dev/null; then
    python_exe=python3
fi

# Create virtual environment
$python_exe -m venv .venv

# Detect OS to choose activation method
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash or Cygwin)
    source .venv/Scripts/activate
else
    # Linux/macOS
    source .venv/bin/activate
    export LDFLAGS="-I/usr/local/opt/openssl/include -L/usr/local/opt/openssl/lib"
fi

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install local src/ package in editable mode
cd src/
pip install -e .
