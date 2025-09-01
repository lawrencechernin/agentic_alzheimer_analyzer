#!/usr/bin/env python3
"""
ADNI Downloader using Environment Variables
Usage: 
export ADNI_USERNAME="your_email"
export ADNI_PASSWORD="your_password" 
python adni_download_env.py
"""

import os
import sys
from pathlib import Path

# Import our fixed downloader
try:
    from adni_downloader_fixed import ADNIDownloaderFixed
except ImportError:
    print("Error: Could not import ADNIDownloaderFixed")
    sys.exit(1)

def main():
    # Get credentials from environment variables
    username = os.getenv('ADNI_USERNAME')
    password = os.getenv('ADNI_PASSWORD')
    
    if not username or not password:
        print("❌ Error: ADNI credentials not found in environment variables")
        print("\nTo set them, run:")
        print('export ADNI_USERNAME="your_email@domain.com"')
        print('export ADNI_PASSWORD="your_password"')
        print("python adni_download_env.py")
        return False
    
    print("\n" + "="*60)
    print("ADNI MEMTRAX MULTIMODAL DATA DOWNLOADER")
    print("="*60)
    print(f"Username: {username}")
    print(f"Download directory: ./adni_downloads")
    print("="*60)
    
    downloader = None
    
    try:
        print("\n🚀 Initializing downloader...")
        downloader = ADNIDownloaderFixed(headless=False)
        
        print("🔐 Logging into ADNI...")
        downloader.login(username=username, password=password)
        
        print("📂 Navigating to download section...")
        if not downloader.navigate_to_downloads():
            raise Exception("Could not access download section")
        
        print("📊 Downloading MemTrax data...")
        success = downloader.download_memtrax_data()
        
        if success:
            print("\n✅ DOWNLOAD COMPLETED!")
            
            # List downloaded files
            files = list(Path("./adni_downloads").glob("*.csv"))
            print(f"📁 Downloaded {len(files)} files to: ./adni_downloads")
            
            for file in files:
                print(f"  • {file.name}")
                
            print(f"\n📈 Next step: Merge the data")
            print(f"cd ..")
            print(f"python scripts/data_merger.py")
        else:
            print("⚠️  No MemTrax data was downloaded")
        
    except Exception as e:
        print(f"\n❌ DOWNLOAD FAILED: {e}")
        return False
        
    finally:
        if downloader:
            print("\n🔄 Closing browser...")
            downloader.close()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)