import os
import platform
import subprocess
import sys
import shutil

def main():
    print("Installing PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    print("Building executable for", platform.system(), "...")
    
    # We want a one-file executable that doesn't pop up a console window
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir", # Onedir is usually better for large apps like PyQt/Tensorflow, but onedir is fine too. Let's use --onedir for faster startup, or --onefile. Let's stick to --onedir for now to avoid the extraction overhead.
        "--windowed", # No console window
        "--name", "ProjectAnvaya",
        "run_app.py"
    ]
    
    # Exclude bloated modules if they aren't strictly needed for the runtime
    # But since tensorflow, mediapipe etc are needed, we can't exclude them.
    # We might want to exclude some things to reduce size if possible.
    cmd.extend([
        "--exclude-module", "pytest",
        "--exclude-module", "pylint",
    ])
    
    # Add data files (QSS, etc.)
    # Format for add-data: "source;destination" on Windows, "source:destination" on Unix
    sep = ";" if platform.system() == "Windows" else ":"
    
    cmd.extend([
        "--add-data", f"style.qss{sep}.",
        "--add-data", f"style_dark.qss{sep}."
    ])

    subprocess.run(cmd, check=True)
    
    print("Build complete! Check the 'dist' folder.")

if __name__ == "__main__":
    main()
