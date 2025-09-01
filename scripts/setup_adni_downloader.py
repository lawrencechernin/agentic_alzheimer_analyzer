#!/usr/bin/env python3
"""
Setup script for ADNI downloader
Installs required dependencies and sets up Chrome WebDriver
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    requirements_file = Path(__file__).parent / "requirements_adni.txt"
    
    print("Installing Python requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
    ], check=True)
    
    print("Installing webdriver-manager...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "webdriver-manager"
    ], check=True)

def setup_chrome_driver():
    """Set up Chrome WebDriver automatically"""
    try:
        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        print("Setting up Chrome WebDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        driver.quit()
        print("Chrome WebDriver setup complete!")
        
    except Exception as e:
        print(f"Chrome WebDriver setup failed: {e}")
        print("Please install Chrome browser and try again")

def create_config_example():
    """Create example configuration file"""
    config_content = '''# ADNI Downloader Configuration
# Copy this to config.py and fill in your details

# ADNI Credentials (optional - script will prompt if not provided)
ADNI_USERNAME = ""
ADNI_PASSWORD = ""

# Download settings
DOWNLOAD_DIR = "./adni_downloads"
HEADLESS_MODE = False  # Set to True to run without browser window

# Tables to download
MEMTRAX_TABLES = [
    "BHR_MEMTRAX",
    "BHR_MEMTRAX_SCORES",
    "REGISTRY",
    "ECOG"
]

# Chrome options
CHROME_OPTIONS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu"
]
'''
    
    config_file = Path(__file__).parent / "config_example.py"
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print(f"Created example config file: {config_file}")

def main():
    """Main setup function"""
    print("Setting up ADNI Downloader...")
    
    try:
        install_requirements()
        setup_chrome_driver()
        create_config_example()
        
        print("\nSetup complete!")
        print("\nNext steps:")
        print("1. Copy config_example.py to config.py")
        print("2. Fill in your ADNI credentials (optional)")
        print("3. Run: python adni_downloader.py")
        print("\nImportant: Ensure you have proper ADNI data access permissions!")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()