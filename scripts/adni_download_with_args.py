#!/usr/bin/env python3
"""
ADNI Downloader with Command Line Arguments
Usage: python adni_download_with_args.py --username your_email --password your_password
"""

import argparse
import sys
import os
from pathlib import Path

# Import our fixed downloader
try:
    from adni_downloader_fixed import ADNIDownloaderFixed
except ImportError:
    print("Error: Could not import ADNIDownloaderFixed")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download ADNI data with credentials')
    parser.add_argument('--username', '-u', required=True, help='ADNI username/email')
    parser.add_argument('--password', '-p', required=True, help='ADNI password')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--download-dir', default='./adni_downloads', help='Download directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ADNI MEMTRAX MULTIMODAL DATA DOWNLOADER")
    print("="*60)
    print(f"Username: {args.username}")
    print(f"Download directory: {args.download_dir}")
    print(f"Headless mode: {args.headless}")
    print("="*60)
    
    downloader = None
    
    try:
        print("\n🚀 Initializing downloader...")
        downloader = ADNIDownloaderFixed(
            download_dir=args.download_dir, 
            headless=args.headless
        )
        
        print("🔐 Logging into ADNI...")
        downloader.login(username=args.username, password=args.password)
        
        print("📂 Navigating to download section...")
        if not downloader.navigate_to_downloads():
            raise Exception("Could not access download section")
        
        print("📊 Downloading MemTrax data...")
        success = downloader.download_memtrax_data()
        
        if success:
            print("\n✅ DOWNLOAD COMPLETED!")
            
            # List downloaded files
            files = list(Path(args.download_dir).glob("*.csv"))
            print(f"📁 Downloaded {len(files)} files to: {args.download_dir}")
            
            for file in files:
                print(f"  • {file.name}")
                
            print(f"\n📈 Next step: Merge the data")
            print(f"python data_merger.py")
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