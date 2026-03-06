import subprocess
import sys
import os

def main():
    """Launch the Streamlit application from the root directory."""
    # Ensure the current directory is in the PYTHONPATH without overwriting existing entries
    env = os.environ.copy()
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{os.getcwd()}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = os.getcwd()
    
    # Security: This call is SAFE. We use a list of arguments and shell=False (default).
    # All components are internally defined or constructed via os.path.join.
    cmd = [
        "streamlit", 
        "run", 
        os.path.join("src", "streamlit_app.py")
    ]
    
    try:
        # Launching with a list of arguments prevents shell injection.
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
