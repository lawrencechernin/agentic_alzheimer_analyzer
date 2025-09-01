#!/usr/bin/env python3
"""
Debug ADNI Login Issues
Test the login process step by step to identify issues
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_login():
    """Debug the ADNI login process"""
    
    # Chrome options for debugging
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Don't run headless so we can see what's happening
    
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 30)
    
    try:
        logger.info("1. Navigating to ADNI login page...")
        driver.get("https://ida.loni.usc.edu/login.jsp")
        
        # Wait a bit for page to load
        time.sleep(5)
        
        logger.info(f"2. Current URL: {driver.current_url}")
        logger.info(f"3. Page title: {driver.title}")
        
        # Check if we're on the right page
        if "login" not in driver.current_url.lower():
            logger.warning("Not on login page! Checking for redirects...")
        
        # Try to find the page source
        logger.info("4. Looking for login form elements...")
        
        # Try different possible login field names
        login_field_names = ['userLogin', 'username', 'user', 'login', 'email']
        password_field_names = ['userPassword', 'password', 'pass', 'pwd']
        
        login_field = None
        password_field = None
        
        # Search for login field
        for name in login_field_names:
            try:
                login_field = driver.find_element(By.NAME, name)
                logger.info(f"✓ Found login field: {name}")
                break
            except NoSuchElementException:
                logger.info(f"✗ Login field not found: {name}")
                continue
        
        # Search for password field  
        for name in password_field_names:
            try:
                password_field = driver.find_element(By.NAME, name)
                logger.info(f"✓ Found password field: {name}")
                break
            except NoSuchElementException:
                logger.info(f"✗ Password field not found: {name}")
                continue
        
        # Look for submit buttons
        submit_selectors = [
            "//input[@type='submit']",
            "//button[@type='submit']", 
            "//input[@value='Login']",
            "//button[contains(text(),'Login')]",
            "//input[contains(@value,'login')]"
        ]
        
        submit_button = None
        for selector in submit_selectors:
            try:
                submit_button = driver.find_element(By.XPATH, selector)
                logger.info(f"✓ Found submit button: {selector}")
                break
            except NoSuchElementException:
                logger.info(f"✗ Submit button not found: {selector}")
                continue
        
        # Print all form elements
        logger.info("5. All form elements on page:")
        forms = driver.find_elements(By.TAG_NAME, "form")
        for i, form in enumerate(forms):
            logger.info(f"Form {i+1}:")
            inputs = form.find_elements(By.TAG_NAME, "input")
            for inp in inputs:
                logger.info(f"  Input: type={inp.get_attribute('type')}, name={inp.get_attribute('name')}, id={inp.get_attribute('id')}")
        
        # Try to get credentials and test login if fields found
        if login_field and password_field and submit_button:
            logger.info("6. All required fields found! Testing login...")
            
            username = input("Enter ADNI username: ")
            password = input("Enter ADNI password: ")
            
            login_field.clear()
            login_field.send_keys(username)
            
            password_field.clear()
            password_field.send_keys(password)
            
            logger.info("7. Submitting login form...")
            submit_button.click()
            
            # Wait and see what happens
            time.sleep(10)
            
            logger.info(f"8. After login - URL: {driver.current_url}")
            logger.info(f"9. After login - Title: {driver.title}")
            
            # Check for success/error messages
            try:
                error_elements = driver.find_elements(By.XPATH, "//*[contains(text(),'error') or contains(text(),'Error') or contains(text(),'invalid') or contains(text(),'failed')]")
                if error_elements:
                    logger.error(f"Login error found: {error_elements[0].text}")
                else:
                    logger.info("No obvious error messages found")
            except:
                pass
                
        else:
            logger.error("Could not find all required login elements!")
            logger.error(f"Login field found: {login_field is not None}")
            logger.error(f"Password field found: {password_field is not None}")
            logger.error(f"Submit button found: {submit_button is not None}")
        
        # Keep browser open for manual inspection
        input("Press Enter to close browser...")
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        
    finally:
        driver.quit()


if __name__ == "__main__":
    debug_login()