#!/bin/bash
echo "Starting Project Anvaya..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
python3 run_app.py || python run_app.py
echo ""
echo "Application has exited or crashed."
read -p "Press Enter to continue..."
