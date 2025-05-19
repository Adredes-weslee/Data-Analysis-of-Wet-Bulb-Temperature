"""
Environment Verification Script
==============================
This script verifies that the environment is properly set up for the wet bulb temperature analysis.
It checks Python version, required packages, and directory structure to ensure the project
can run correctly.

Usage: 
    python verify_environment.py

The script will report on:
1. Python version compatibility
2. Required package installation status
3. Project directory structure
4. Availability of required data files
"""
import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """
    Check if Python version meets requirements
    
    Verifies that the current Python version is compatible with the project
    requirements. The project was developed with Python 3.8+, so this function
    warns if an older version is being used.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
    """
    print(f"Python version: {sys.version}")
    major, minor, *_ = sys.version_info
    if major != 3 or minor < 8:
        print("⚠️ Warning: This project was developed with Python 3.8+. Some features may not work correctly.")
    else:
        print("✅ Python version: OK")

def check_required_packages():
    """
    Check if required packages are installed
    
    Attempts to import each required package and reports on whether
    it is installed and its version. Provides instructions for installing
    missing packages if any are found.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 
        'statsmodels', 'streamlit', 'jupyter', 'nbformat'
    ]
    
    missing_packages = []
    outdated_packages = []
    
    print("\nChecking required packages:")
    for package in required_packages:
        if package == 'jupyter':
            try:
                notebook_module = importlib.import_module('notebook')
                version = getattr(notebook_module, '__version__', 'unknown')
                print(f"✅ {package}: {version}")
            except ImportError:
                print(f"❌ {package}: Not installed")
                missing_packages.append(package)
            continue
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Missing packages:")
        print("Run the following command to install them:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required packages are installed")

def check_directory_structure():
    """
    Check if the directory structure is correct
    
    Verifies that all required directories for the project exist.
    This ensures that the code can find necessary files and save
    outputs in the expected locations. Missing directories will be
    created automatically.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
        
    Side Effects
    -----------
    Creates missing directories if they don't exist
    """
    # Define expected directories
    project_root = Path(__file__).parent.parent
    expected_dirs = [
        'data/raw', 'data/processed', 'data/output',
        'src/app_pages', 'src/data_processing', 'src/features', 
        'src/models', 'src/utils', 'src/visualization',
        'dashboard', 'notebooks'
    ]
    
    print("\nChecking directory structure:")
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}: OK")
        else:
            print(f"❌ {dir_path}: Missing")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  📂 Created {dir_path}")
            except Exception as e:
                print(f"  ⚠️ Could not create {dir_path}: {e}")

def check_data_files():
    """
    Check if the necessary data files exist
    
    Verifies that all required data files for the project are present
    in the expected directory. This ensures that the analysis can be
    performed without missing inputs.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
        
    Raises
    ------
    FileNotFoundError
        Indirectly if the data directory doesn't exist (when trying to check files)
    """
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    required_files = [
        'wet-bulb-temperature-hourly.csv', 
        'surface-air-temperature-monthly-mean.csv',
        'M890081.csv',
        'co2_mm_mlo.csv',
        'ch4_mm_gl.csv',
        'n2o_mm_gl.csv',
        'sf6_mm_gl.csv'
    ]
    
    print("\nChecking raw data files:")
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file}: OK")
        else:
            print(f"❌ {file}: Missing")

def check_streamlit():
    """
    Check if Streamlit is working correctly
    
    Verifies that Streamlit is installed and can be executed. This ensures
    that the dashboard functionality of the project is operational.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
    """
    print("\nChecking Streamlit installation:")
    try:
        result = subprocess.run(
            ['streamlit', '--version'], 
            capture_output=True, 
            text=True
        )
        print(f"✅ Streamlit is installed: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Error checking Streamlit: {e}")

def main():
    """
    Main function to run all checks
    
    Executes all verification functions in sequence to ensure the environment
    is properly set up for the wet bulb temperature analysis project.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Prints status messages to the console
    """
    print("=" * 50)
    print("Wet Bulb Temperature Analysis - Environment Verification")
    print("=" * 50)
    
    check_python_version()
    check_required_packages()
    check_directory_structure()
    check_data_files()
    check_streamlit()
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
