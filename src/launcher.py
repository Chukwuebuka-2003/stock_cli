import sys
import os
from streamlit.web import cli as stcli

def main():
    """Launch the Streamlit application."""
    # Get the directory of this file (src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to streamlit_app.py
    app_path = os.path.join(src_dir, "streamlit_app.py")
    
    # Check if file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find streamlit_app.py at {app_path}")
        sys.exit(1)

    # Construct the command arguments for streamlit
    # This simulates running: streamlit run /path/to/streamlit_app.py
    sys.argv = ["streamlit", "run", app_path]

    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
