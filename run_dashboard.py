"""
Main module for running the Streamlit dashboard application

This script serves as the entry point for launching the Streamlit dashboard
application. It sets up the Python path to include the project root directory
and then launches the Streamlit server with the dashboard application.

Usage:
    python run_dashboard.py

The dashboard will be accessible through a web browser, typically at
http://localhost:8501 unless another port is specified.

Side Effects:
    - Launches a Streamlit server process
    - Opens a web browser window (behavior depends on Streamlit configuration)
    - Creates a .streamlit directory for caching if it doesn't exist

Notes:
    This script is a convenience wrapper around the 'streamlit run' command.
    It ensures that the Python path is set up correctly so that the dashboard
    application can find all required modules.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run the Streamlit app
if __name__ == "__main__":
    os.system("streamlit run dashboard/app.py")
