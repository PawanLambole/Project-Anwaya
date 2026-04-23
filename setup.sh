#!/bin/bash
echo "Setting up Project Anvaya..."
python3 -m venv venv || python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete! You can now run ./start_app.sh"
