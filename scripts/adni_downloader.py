#!/usr/bin/env python3
"""
ADNI IDA Data Downloader
Automated download of ADNI data using Selenium web automation

IMPORTANT: This script is for educational purposes and should only be used
in compliance with ADNI's terms of service and data use agreement.
"""

import os
import time
import getpass
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADNIDownloader:
    def __init__(self, download_dir=None, headless=False):
        """
        Initialize ADNI downloader
        
        Args:
            download_dir: Directory to save downloads (default: ./adni_downloads)
            headless: Run browser in headless mode
        """
        self.download_dir = Path(download_dir) if download_dir else Path("./adni_downloads")
        self.download_dir.mkdir(exist_ok=True)
        
        # Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set download directory
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 30)
        
        logger.info(f"Download directory: {self.download_dir}")
    
    def login(self, username=None, password=None):
        """
        Login to ADNI IDA portal
        
        Args:
            username: ADNI username (will prompt if not provided)
            password: ADNI password (will prompt if not provided)
        """
        if not username:
            username = input("ADNI Username: ")
        if not password:
            password = getpass.getpass("ADNI Password: ")
        
        logger.info("Navigating to ADNI login page...")
        self.driver.get("https://ida.loni.usc.edu/login.jsp")
        
        try:
            # Wait for login form
            username_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "userLogin"))
            )
            password_field = self.driver.find_element(By.NAME, "userPassword")
            login_button = self.driver.find_element(By.XPATH, "//input[@type='submit'][@value='Login']")
            
            # Enter credentials
            username_field.clear()
            username_field.send_keys(username)
            password_field.clear()
            password_field.send_keys(password)
            
            logger.info("Submitting login credentials...")
            login_button.click()
            
            # Wait for successful login (check for dashboard/home page)
            self.wait.until(
                EC.url_contains("pages/access")
            )
            logger.info("Successfully logged in to ADNI IDA")
            
        except TimeoutException:
            logger.error("Login failed or took too long")
            raise
    
    def navigate_to_downloads(self):
        """Navigate to the downloads/data section"""
        try:
            logger.info("Navigating to data downloads...")
            
            # Look for Download or Data menu
            download_link = self.wait.until(
                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Download"))
            )
            download_link.click()
            
            # Wait for download page to load
            time.sleep(3)
            
        except TimeoutException:
            logger.error("Could not find download section")
            raise
    
    def search_data(self, search_terms=None):
        """
        Search for specific data tables
        
        Args:
            search_terms: List of search terms (e.g., ['BHR_MEMTRAX', 'ECOG'])
        """
        if not search_terms:
            search_terms = ['BHR_MEMTRAX']
        
        logger.info(f"Searching for data: {search_terms}")
        
        try:
            # Look for search box or advanced search
            search_box = self.wait.until(
                EC.presence_of_element_located((By.NAME, "searchTerm"))
            )
            
            for term in search_terms:
                search_box.clear()
                search_box.send_keys(term)
                
                # Submit search
                search_button = self.driver.find_element(By.XPATH, "//input[@type='submit'][@value='Search']")
                search_button.click()
                
                time.sleep(2)
                logger.info(f"Searched for: {term}")
                
        except NoSuchElementException:
            logger.error("Could not find search interface")
            raise
    
    def download_table(self, table_name):
        """
        Download a specific data table
        
        Args:
            table_name: Name of the table to download (e.g., 'BHR_MEMTRAX')
        """
        logger.info(f"Attempting to download table: {table_name}")
        
        try:
            # Look for table in results
            table_link = self.wait.until(
                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, table_name))
            )
            table_link.click()
            
            time.sleep(2)
            
            # Look for download button/link
            download_options = [
                "Download",
                "Export",
                "CSV",
                "Download CSV"
            ]
            
            for option in download_options:
                try:
                    download_btn = self.driver.find_element(By.PARTIAL_LINK_TEXT, option)
                    download_btn.click()
                    logger.info(f"Initiated download for {table_name}")
                    time.sleep(5)  # Wait for download to start
                    break
                except NoSuchElementException:
                    continue
            else:
                logger.warning(f"Could not find download option for {table_name}")
                
        except TimeoutException:
            logger.error(f"Could not find table: {table_name}")
    
    def download_memtrax_data(self):
        """Specifically download MemTrax-related data"""
        memtrax_tables = [
            'BHR_MEMTRAX',
            'BHR_MEMTRAX_SCORES',  # If it exists
            'MEMTRAX',
            'REGISTRY'  # For subject info
        ]
        
        logger.info("Downloading MemTrax data...")
        
        for table in memtrax_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(3)
            except Exception as e:
                logger.warning(f"Could not download {table}: {e}")
                continue
    
    def download_neuroimaging_data(self):
        """Download MRI and PET data for MemTrax participants"""
        
        # MRI Data Tables
        mri_tables = [
            'MRI3META',           # MRI metadata
            'MRILIST',            # MRI scan list
            'MPRAGE',             # T1-weighted structural MRI
            'MPRAGEMETA',         # MPRAGE metadata
            'UCSFFSL',            # FreeSurfer volumetric data
            'UCSFFSL_02_01_16',   # Updated FreeSurfer data
            'UCSFVOL',            # UCSF volumetric measurements
            'DTIROI',             # DTI region of interest data
        ]
        
        # PET Data Tables  
        pet_tables = [
            'PETLIST',            # PET scan list
            'PETMETA',            # PET metadata
            'PIB',                # Pittsburgh Compound B PET
            'PIBMETA',            # PIB metadata
            'AV45',               # Florbetapir PET
            'AV45META',           # AV45 metadata
            'AV1451',             # Flortaucipir (tau) PET
            'AV1451META',         # AV1451 metadata
            'FDG',                # FDG-PET glucose metabolism
            'FDGMETA',            # FDG metadata
            'SUMMARYSUVR',        # PET SUVR summary data
            'BAIPETNMRC',         # Processed PET data
            'UCBERKELEYAV45',     # UC Berkeley AV45 processing
            'UCBERKELEYAV1451',   # UC Berkeley AV1451 processing
        ]
        
        logger.info("Downloading MRI data...")
        for table in mri_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(3)
            except Exception as e:
                logger.warning(f"Could not download MRI table {table}: {e}")
                continue
        
        logger.info("Downloading PET data...")
        for table in pet_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(3)
            except Exception as e:
                logger.warning(f"Could not download PET table {table}: {e}")
                continue
    
    def download_biomarker_data(self):
        """Download blood/plasma and CSF biomarker data"""
        
        # Blood/Plasma biomarker tables
        blood_biomarker_tables = [
            'PLASMA',                    # General plasma biomarkers
            'PLASMA_ABETA',             # Plasma amyloid-beta
            'PLASMA_TAU',               # Plasma tau
            'PLASMA_PTAU',              # Plasma phospho-tau
            'PLASMA_NFL',               # Plasma neurofilament light
            'PLASMA_GFAP',              # Plasma glial fibrillary acidic protein
            'SIMOA',                    # Simoa plasma biomarkers
            'SIMOA_ABETA',              # Simoa amyloid measurements
            'SIMOA_TAU',                # Simoa tau measurements
            'SIMOA_NFL',                # Simoa NFL measurements
            'ELECSYS',                  # Elecsys platform biomarkers
            'ELECSYS_ABETA',            # Elecsys amyloid
            'ELECSYS_TAU',              # Elecsys tau
            'ELECSYS_PTAU',             # Elecsys phospho-tau
            'LUMIPULSE',                # Lumipulse platform
            'LUMIPULSE_ABETA',          # Lumipulse amyloid
            'LUMIPULSE_TAU',            # Lumipulse tau
            'C2N_PLASMA',               # C2N plasma biomarkers
            'PLASMA_PROTEOMICS',        # Plasma proteomics
            'OLINK',                    # Olink proteomics platform
            'SOMALOGIC',                # SomaLogic proteomics
        ]
        
        # CSF biomarker tables (for comparison)
        csf_biomarker_tables = [
            'UPENNBIOMK9_04_19_17',     # Penn CSF biomarkers
            'UPENNBIOMK_MASTER',        # Master CSF file
            'CSF_ABETA',                # CSF amyloid-beta
            'CSF_TAU',                  # CSF total tau
            'CSF_PTAU',                 # CSF phospho-tau
            'CSF_NFL',                  # CSF neurofilament light
            'CSF_PROTEOMICS',           # CSF proteomics
            'BIOFINDER',                # BioFINDER CSF data
        ]
        
        # Additional biomarker-related tables
        biomarker_support_tables = [
            'BIOMARK',                  # General biomarker table
            'SPECIMEN',                 # Specimen collection info
            'BIOFLUID',                 # Biofluid collection details
            'LABDATA',                  # Laboratory data
            'APOERES',                  # APOE genotyping
            'GWAS',                     # Genome-wide association data
            'WGS',                      # Whole genome sequencing
            'PROTEOMICS_METADATA',      # Proteomics metadata
        ]
        
        logger.info("Downloading blood/plasma biomarker data...")
        for table in blood_biomarker_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not download blood biomarker table {table}: {e}")
                continue
        
        logger.info("Downloading CSF biomarker data...")
        for table in csf_biomarker_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not download CSF biomarker table {table}: {e}")
                continue
        
        logger.info("Downloading biomarker support data...")
        for table in biomarker_support_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not download biomarker support table {table}: {e}")
                continue
    
    def download_comprehensive_dataset(self):
        """Download comprehensive dataset including MemTrax, MRI, PET, blood biomarkers, and clinical data"""
        
        # Clinical and demographic data
        clinical_tables = [
            'PTDEMOG',            # Demographics
            'ADNIMERGE',          # Master merge file with key variables
            'CDR',                # Clinical Dementia Rating
            'ECOG',               # Everyday Cognition scale
            'MMSE',               # Mini-Mental State Exam
            'ADAS',               # Alzheimer's Disease Assessment Scale
            'MOCA',               # Montreal Cognitive Assessment
            'FAQ',                # Functional Activities Questionnaire
            'GDS',                # Geriatric Depression Scale
            'NPI',                # Neuropsychiatric Inventory
            'VISITS',             # Visit information
            'ARM',                # Study arm assignments
            'DXSUM_PDXCONV_ADNIALL',  # Diagnosis summaries
        ]
        
        logger.info("Starting comprehensive multimodal dataset download...")
        logger.info("This will download MemTrax + MRI + PET + Blood Biomarkers + Clinical data")
        
        # Download in logical order
        logger.info("1. Downloading MemTrax cognitive data...")
        self.download_memtrax_data()
        
        logger.info("2. Downloading clinical and demographic data...")
        for table in clinical_tables:
            try:
                self.search_data([table])
                self.download_table(table)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not download clinical table {table}: {e}")
                continue
        
        logger.info("3. Downloading neuroimaging data (MRI + PET)...")
        self.download_neuroimaging_data()
        
        logger.info("4. Downloading blood and CSF biomarker data...")
        self.download_biomarker_data()
        
        logger.info("Comprehensive dataset download completed!")
    
    def list_downloaded_files(self):
        """List files in download directory"""
        files = list(self.download_dir.glob("*"))
        logger.info(f"Downloaded files ({len(files)}):")
        for file in files:
            logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        return files
    
    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("Browser closed")


def main():
    """Main function to demonstrate usage"""
    downloader = None
    
    try:
        # Initialize downloader
        downloader = ADNIDownloader(headless=False)  # Set to True for headless mode
        
        # Login (will prompt for credentials)
        downloader.login()
        
        # Navigate to downloads
        downloader.navigate_to_downloads()
        
        # Download comprehensive multimodal dataset
        # This includes: MemTrax + MRI + PET + Blood Biomarkers + Clinical data
        downloader.download_comprehensive_dataset()
        
        # Alternative: Download just specific components
        # downloader.download_memtrax_data()
        # downloader.download_neuroimaging_data()
        # downloader.download_biomarker_data()
        
        # List downloaded files
        downloader.list_downloaded_files()
        
        logger.info("Download process completed")
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        
    finally:
        if downloader:
            downloader.close()


if __name__ == "__main__":
    main()