import subprocess
import sys
import os

def main():
    """Launch the Streamlit application from the root directory."""
    # Ensure the current directory is in the PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        "streamlit", 
        "run", 
        os.path.join("src", "streamlit_app.py")
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
