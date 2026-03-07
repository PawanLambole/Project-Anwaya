import os
import sys
import time
import subprocess

def get_latest_mtime():
    """Gets the latest modification time of any .py or .qss file in the directory."""
    latest = 0
    for root, dirs, files in os.walk('.'):
        # Skip irrelevant directories
        if '__pycache__' in root or '.git' in root or 'venv' in root or 'ISL_Processed' in root or 'ISL_Data' in root or 'model' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.qss'):
                path = os.path.join(root, file)
                try:
                    mtime = os.stat(path).st_mtime
                    if mtime > latest:
                        latest = mtime
                except Exception:
                    pass
    return latest

def main():
    print("--- 🚀 Starting Auto-Reload Development Server ---")
    print("Watching for file changes (.py, .qss)...")
    
    process = None
    last_mtime = get_latest_mtime()
    
    def start_process():
        # Using sys.executable to use the same python environment
        return subprocess.Popen([sys.executable, 'run_app.py'])
        
    process = start_process()
    
    try:
        while True:
            time.sleep(1) # Check every second
            current_mtime = get_latest_mtime()
            if current_mtime > last_mtime:
                print("\n[DEV] File change detected. Restarting application...\n")
                
                # Terminate the current process gracefully
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill() # Force kill if it hangs
                
                last_mtime = current_mtime
                process = start_process()
                
            # If the process crashed or was closed manually, restart it if we change a file
            if process and process.poll() is not None:
                # We do not auto-restart on crash unless a file is changed to avoid infinite crash loops
                pass

    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
        print("\nDevelopment server stopped.")

if __name__ == '__main__':
    main()
