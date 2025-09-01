# ADNI Downloader Configuration
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
