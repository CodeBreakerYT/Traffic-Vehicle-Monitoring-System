import os
import sys
import subprocess

def main():
    print("=========================================================")
    print("             COMPILING CAMERA STREAMER TO EXE            ")
    print("=========================================================")

    # 1. Ensure we are in the correct directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print(f"[INFO] Working directory set to: {current_dir}")

    # 2. Check and install pyinstaller if missing
    try:
        import PyInstaller
        print("[INFO] PyInstaller is already installed.")
    except ImportError:
        print("[INFO] PyInstaller not detected in current environment. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[INFO] PyInstaller installed successfully.")

    # 3. Define pyinstaller command arguments
    # We build a single file executable named 'cam' and place it in the assets/export directory.
    script_path = "camera_streamer.py"
    dist_path = os.path.abspath(os.path.join(current_dir, "..", "..", "assets", "export"))
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name=cam",
        f"--distpath={dist_path}",
        "--workpath=build",
        "--specpath=.",
        script_path
    ]

    print(f"\n[EXEC] Running command: {' '.join(cmd)}")
    
    try:
        # Run compilation
        subprocess.check_call(cmd)
        print("\n=========================================================")
        print(" SUCCESS: Compiling finished successfully!")
        print(" Executable 'cam.exe' is available in the 'assets/export' folder.")
        print("=========================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL] Compilation failed with error code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL] Unexpected error during compilation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
