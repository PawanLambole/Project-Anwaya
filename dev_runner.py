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

def kill_orphans():
    """Kills any orphaned run_app.py processes to avoid multi-window bugs."""
    try:
        import psutil
        current_pid = os.getpid()
        killed_any = False
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = p.info.get('cmdline') or []
                name = p.info.get('name', '').lower()
                
                # Match "python.exe", "python", "pythonw.exe", etc.
                if 'python' in name:
                    cmd_str = ' '.join([str(c).lower() for c in cmdline])
                    # Only kill orphaned app windows; never kill dev_runner itself.
                    # dev_runner being killed here can terminate the current session.
                    if p.info['pid'] != current_pid and ('run_app.py' in cmd_str):
                        print(f"[Cleanup] Killing orphaned process PID {p.info['pid']} -> {cmd_str}")
                        p.kill()
                        killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError):
                pass
        
        if killed_any:
            import time
            time.sleep(1.0) # Give Windows a second to release camera resources
            
    except ImportError:
        print("[Cleanup] psutil not installed, skipping orphan cleanup.")

def main():
    kill_orphans()
    print("--- Starting Auto-Reload Development Server ---")
    print("Watching for file changes (.py, .qss)...")
    
    process = None
    last_mtime = get_latest_mtime()
    
    def start_process():
        # Always use the virtual environment's python if it exists, 
        # so PyQt5 and Tensorflow are actually found when run_app.py is launched
        venv_python = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')
        if os.path.exists(venv_python):
            python_exe = venv_python
        else:
            python_exe = sys.executable
        return subprocess.Popen([python_exe, 'run_app.py'])
        
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
