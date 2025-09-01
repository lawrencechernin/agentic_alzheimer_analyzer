#!/usr/bin/env python3
"""
ADNI IDA Data Downloader - Fixed Version
Updated with better login handling and error detection
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

class ADNIDownloaderFixed:
    def __init__(self, download_dir=None, headless=False):
        """
        Initialize ADNI downloader with improved error handling
        """
        self.download_dir = Path(download_dir) if download_dir else Path("./adni_downloads")
        self.download_dir.mkdir(exist_ok=True)
        
        # Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set download directory
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 45)  # Increased timeout
        
        logger.info(f"Download directory: {self.download_dir}")
    
    def check_page_loaded(self, expected_elements=None):
        """Check if page has loaded properly"""
        try:
            # Wait for body to be present
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)  # Additional wait for dynamic content
            
            logger.info(f"Page loaded: {self.driver.title}")
            logger.info(f"Current URL: {self.driver.current_url}")
            
            # Handle cookie banner/popup
            self.handle_cookie_banner()
            
            return True
        except TimeoutException:
            logger.error("Page failed to load properly")
            return False
    
    def handle_cookie_banner(self):
        """Handle cookie acceptance banner"""
        cookie_selectors = [
            "//button[contains(text(),'Accept')]",
            "//button[contains(text(),'OK')]", 
            "//button[contains(text(),'I agree')]",
            "//button[contains(text(),'Continue')]",
            "//a[contains(text(),'Accept')]",
            "//div[contains(@class,'cookie')]//button",
            "//div[contains(@class,'banner')]//button",
            "//button[contains(@class,'accept')]",
            "//button[contains(@class,'cookie')]",
            "//*[@id='cookieAccept']",
            "//*[@id='cookie-accept']",
        ]
        
        for selector in cookie_selectors:
            try:
                cookie_button = self.driver.find_element(By.XPATH, selector)
                if cookie_button.is_displayed():
                    cookie_button.click()
                    logger.info("✓ Accepted cookies")
                    time.sleep(2)
                    return True
            except NoSuchElementException:
                continue
        
        logger.info("No cookie banner found or already handled")
        return False
    
    def find_login_elements(self):
        """Try multiple strategies to find login elements"""
        login_strategies = [
            # Strategy 1: Correct ADNI field names (from debug)
            {'login': 'userEmail', 'password': 'userPassword'},
            # Strategy 2: Backup ADNI names
            {'login': 'userLogin', 'password': 'userPassword'},
            # Strategy 3: Generic names
            {'login': 'username', 'password': 'password'},
            {'login': 'user', 'password': 'pass'},
            # Strategy 4: By ID
            {'login': 'login', 'password': 'password'},
        ]
        
        for i, strategy in enumerate(login_strategies):
            logger.info(f"Trying login strategy {i+1}: {strategy}")
            try:
                login_field = self.driver.find_element(By.NAME, strategy['login'])
                password_field = self.driver.find_element(By.NAME, strategy['password'])
                logger.info(f"✓ Found login fields with strategy {i+1}")
                return login_field, password_field
            except NoSuchElementException:
                continue
        
        # Strategy 4: Find by input type
        try:
            login_field = self.driver.find_element(By.XPATH, "//input[@type='text' or @type='email']")
            password_field = self.driver.find_element(By.XPATH, "//input[@type='password']")
            logger.info("✓ Found login fields by input type")
            return login_field, password_field
        except NoSuchElementException:
            pass
        
        logger.error("Could not find login fields with any strategy")
        return None, None
    
    def find_submit_button(self):
        """Try multiple strategies to find submit button"""
        submit_strategies = [
            # Look for any submit buttons first
            (By.XPATH, "//input[@type='submit']"),
            (By.XPATH, "//button[@type='submit']"),
            # Look for buttons with login text
            (By.XPATH, "//input[@type='submit'][@value='Login']"),
            (By.XPATH, "//button[@type='submit' and contains(text(),'Login')]"),
            (By.XPATH, "//*[contains(text(),'Login') or contains(text(),'login')][self::button or self::input]"),
            # Look for any clickable element near password field
            (By.XPATH, "//form//input[@type='button']"),
            (By.XPATH, "//form//button"),
            # Sometimes it's a link or div that acts as submit
            (By.XPATH, "//a[contains(text(),'Login') or contains(text(),'login')]"),
            (By.XPATH, "//*[@onclick and (contains(text(),'Login') or contains(text(),'login'))]"),
        ]
        
        for i, (by, selector) in enumerate(submit_strategies):
            try:
                button = self.driver.find_element(by, selector)
                logger.info(f"✓ Found submit button with strategy {i+1}")
                return button
            except NoSuchElementException:
                continue
        
        logger.error("Could not find submit button")
        return None
    
    def login(self, username=None, password=None):
        """
        Improved login method with multiple fallback strategies
        """
        if not username:
            username = input("ADNI Username: ")
        if not password:
            password = getpass.getpass("ADNI Password: ")
        
        logger.info("Navigating to ADNI login page...")
        
        # Try the main login URL first
        login_urls = [
            "https://ida.loni.usc.edu/login.jsp",
            "https://ida.loni.usc.edu/",
            "https://adni.loni.usc.edu/",
        ]
        
        login_successful = False
        
        for url in login_urls:
            try:
                logger.info(f"Trying URL: {url}")
                self.driver.get(url)
                
                if not self.check_page_loaded():
                    continue
                
                # Check if we're already logged in or redirected
                if "login" not in self.driver.current_url.lower() and "home" in self.driver.current_url.lower():
                    logger.info("Already logged in or redirected to home page")
                    return True
                
                # Find login elements
                login_field, password_field = self.find_login_elements()
                if not login_field or not password_field:
                    logger.warning(f"Login fields not found on {url}")
                    continue
                
                # Find submit button (but continue even if not found)
                submit_button = self.find_submit_button()
                if not submit_button:
                    logger.warning(f"Submit button not found on {url}, will use alternative submission")
                    submit_button = None  # We'll handle this below
                
                # Attempt login
                logger.info("Entering credentials...")
                
                # Ensure fields are interactable
                try:
                    # Scroll to login field and ensure it's visible
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", login_field)
                    time.sleep(1)
                    
                    # Wait for field to be clickable
                    login_field = self.wait.until(EC.element_to_be_clickable(login_field))
                    
                    login_field.clear()
                    login_field.send_keys(username)
                    logger.info("✓ Username entered")
                    time.sleep(1)
                    
                    # Same for password field
                    password_field = self.wait.until(EC.element_to_be_clickable(password_field))
                    password_field.clear()
                    password_field.send_keys(password)
                    logger.info("✓ Password entered")
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Could not interact with login fields: {e}")
                    # Try JavaScript injection as fallback
                    try:
                        self.driver.execute_script("arguments[0].value = arguments[1];", login_field, username)
                        self.driver.execute_script("arguments[0].value = arguments[1];", password_field, password)
                        logger.info("✓ Credentials entered via JavaScript")
                    except Exception as e2:
                        logger.error(f"JavaScript fallback failed: {e2}")
                        continue
                
                logger.info("Submitting form...")
                
                if submit_button:
                    try:
                        submit_button.click()
                        logger.info("Clicked submit button")
                    except Exception as e:
                        logger.warning(f"Submit button click failed: {e}")
                        submit_button = None  # Fall back to alternatives
                
                if not submit_button:
                    # Alternative submission methods when no button found
                    logger.info("Using alternative form submission...")
                    
                    # Method 1: Submit form using JavaScript
                    try:
                        self.driver.execute_script("arguments[0].form.submit();", password_field)
                        logger.info("✓ Submitted form via JavaScript")
                    except Exception as e:
                        logger.warning(f"JavaScript submit failed: {e}")
                        
                        # Method 2: Press Enter in password field
                        try:
                            from selenium.webdriver.common.keys import Keys
                            password_field.send_keys(Keys.RETURN)
                            logger.info("✓ Submitted form via Enter key")
                        except Exception as e2:
                            logger.warning(f"Enter key submit failed: {e2}")
                            
                            # Method 3: Try to find and click any nearby clickable element
                            try:
                                # Look for any element that might trigger login
                                nearby_elements = self.driver.find_elements(By.XPATH, "//form//*[@onclick or @type='button' or contains(@class,'button') or contains(@class,'btn')]")
                                if nearby_elements:
                                    nearby_elements[0].click()
                                    logger.info("✓ Clicked nearby form element")
                                else:
                                    logger.error("All submission methods failed")
                                    continue
                            except Exception as e3:
                                logger.error(f"Final submission attempt failed: {e3}")
                                continue
                
                # Wait for redirect/response
                time.sleep(8)
                
                # Check for successful login
                current_url = self.driver.current_url.lower()
                if any(success_indicator in current_url for success_indicator in ['home', 'dashboard', 'main', 'welcome']):
                    logger.info("Login successful!")
                    login_successful = True
                    break
                elif "login" in current_url:
                    logger.warning("Still on login page - credentials might be incorrect")
                    # Check for error messages
                    self.check_for_error_messages()
                else:
                    logger.info(f"Redirected to: {self.driver.current_url}")
                    # Might be successful, continue
                    login_successful = True
                    break
                    
            except Exception as e:
                logger.error(f"Login attempt failed for {url}: {e}")
                continue
        
        if not login_successful:
            raise Exception("Login failed with all attempted URLs")
        
        return True
    
    def check_for_error_messages(self):
        """Check for error messages on the page"""
        error_selectors = [
            "//*[contains(text(),'error')]",
            "//*[contains(text(),'Error')]", 
            "//*[contains(text(),'invalid')]",
            "//*[contains(text(),'incorrect')]",
            "//*[contains(text(),'failed')]",
            "//*[@class='error']",
            "//*[@class='alert']"
        ]
        
        for selector in error_selectors:
            try:
                errors = self.driver.find_elements(By.XPATH, selector)
                if errors:
                    for error in errors:
                        if error.is_displayed() and error.text.strip():
                            logger.error(f"Error message found: {error.text}")
            except:
                pass
    
    def navigate_to_downloads(self):
        """Navigate to the downloads/data section with improved detection"""
        logger.info("Looking for data download section...")
        
        # Multiple strategies to find download/data section
        download_selectors = [
            "//a[contains(text(),'Download')]",
            "//a[contains(text(),'Data')]",
            "//a[contains(text(),'Study Data')]", 
            "//a[contains(text(),'download')]",
            "//a[contains(text(),'data')]",
            "//*[@href*='download']",
            "//*[@href*='data']"
        ]
        
        for selector in download_selectors:
            try:
                download_link = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                logger.info(f"Found download link: {download_link.text}")
                download_link.click()
                time.sleep(3)
                return True
            except TimeoutException:
                continue
        
        logger.error("Could not find download section")
        return False
    
    def search_and_download_table(self, table_name):
        """Search for and download a specific table with improved error handling"""
        logger.info(f"Searching for table: {table_name}")
        
        try:
            # Look for search functionality
            search_selectors = [
                "//input[@name='searchTerm']",
                "//input[@name='search']",
                "//input[@type='search']",
                "//input[@placeholder*='search']",
                "//input[@placeholder*='Search']"
            ]
            
            search_box = None
            for selector in search_selectors:
                try:
                    search_box = self.driver.find_element(By.XPATH, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if search_box:
                search_box.clear()
                search_box.send_keys(table_name)
                
                # Look for search button
                search_buttons = [
                    "//input[@type='submit'][@value*='Search']",
                    "//button[contains(text(),'Search')]",
                    "//input[@type='submit']"
                ]
                
                for button_selector in search_buttons:
                    try:
                        search_btn = self.driver.find_element(By.XPATH, button_selector)
                        search_btn.click()
                        time.sleep(3)
                        break
                    except NoSuchElementException:
                        continue
            
            # Look for the table in results
            table_selectors = [
                f"//a[contains(text(),'{table_name}')]",
                f"//*[contains(text(),'{table_name}')]//a",
                f"//td[contains(text(),'{table_name}')]"
            ]
            
            for selector in table_selectors:
                try:
                    table_element = self.driver.find_element(By.XPATH, selector)
                    if table_element.tag_name == 'a':
                        table_element.click()
                    else:
                        # Find nearby link
                        parent = table_element.find_element(By.XPATH, "..")
                        link = parent.find_element(By.TAG_NAME, "a")
                        link.click()
                    
                    time.sleep(3)
                    
                    # Look for download options
                    download_options = [
                        "//a[contains(text(),'Download')]",
                        "//a[contains(text(),'CSV')]",
                        "//a[contains(text(),'Export')]",
                        "//input[@value='Download']",
                        "//button[contains(text(),'Download')]"
                    ]
                    
                    for option in download_options:
                        try:
                            download_btn = self.driver.find_element(By.XPATH, option)
                            download_btn.click()
                            logger.info(f"Downloaded {table_name}")
                            time.sleep(5)
                            return True
                        except NoSuchElementException:
                            continue
                    
                    logger.warning(f"Found {table_name} but no download option")
                    return False
                    
                except NoSuchElementException:
                    continue
            
            logger.warning(f"Table {table_name} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error downloading {table_name}: {e}")
            return False
    
    def download_memtrax_data(self):
        """Download MemTrax data with improved error handling"""
        memtrax_tables = ['BHR_MEMTRAX', 'MEMTRAX', 'REGISTRY']
        
        logger.info("Downloading MemTrax data...")
        successful_downloads = 0
        
        for table in memtrax_tables:
            if self.search_and_download_table(table):
                successful_downloads += 1
        
        logger.info(f"Successfully downloaded {successful_downloads}/{len(memtrax_tables)} MemTrax tables")
        return successful_downloads > 0
    
    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("Browser closed")


def main():
    """Main function with improved error handling"""
    downloader = None
    
    try:
        logger.info("Starting ADNI download with improved error handling...")
        
        # Initialize downloader
        downloader = ADNIDownloaderFixed(headless=False)
        
        # Login with better error handling
        downloader.login()
        
        # Navigate to downloads
        if not downloader.navigate_to_downloads():
            raise Exception("Could not navigate to download section")
        
        # Download MemTrax data
        if downloader.download_memtrax_data():
            logger.info("MemTrax data download completed")
        else:
            logger.warning("No MemTrax data was downloaded")
        
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        print("\n" + "="*50)
        print("DOWNLOAD FAILED")
        print("="*50)
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Verify your ADNI credentials are correct")
        print("2. Check that you have active ADNI data access")
        print("3. Ensure Chrome browser is installed and updated")
        print("4. Try running the debug script: python debug_adni_login.py")
        
    finally:
        if downloader:
            downloader.close()


if __name__ == "__main__":
    main()