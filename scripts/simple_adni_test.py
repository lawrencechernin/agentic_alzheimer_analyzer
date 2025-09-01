#!/usr/bin/env python3
"""
Simple ADNI Login Test
Run this directly in your terminal: python simple_adni_test.py
"""

import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

def main():
    username = "lawrence@dabble.health"  # Your username
    password = input("Enter your ADNI password: ")
    
    # Setup Chrome  
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("\n🚀 Testing ADNI login...")
        
        # Go to login page
        driver.get("https://ida.loni.usc.edu/login.jsp")
        time.sleep(5)
        
        print(f"Current URL: {driver.current_url}")
        print(f"Page title: {driver.title}")
        
        # Handle cookies
        try:
            cookie_btn = driver.find_element(By.XPATH, "//button[contains(text(),'Accept')]")
            if cookie_btn.is_displayed():
                cookie_btn.click()
                print("✅ Accepted cookies")
                time.sleep(2)
        except:
            print("No cookie banner")
        
        # Find login fields
        email_field = driver.find_element(By.NAME, "userEmail")
        password_field = driver.find_element(By.NAME, "userPassword")
        
        print("✅ Found login fields")
        
        # Enter credentials using JavaScript (most reliable)
        driver.execute_script("arguments[0].value = arguments[1];", email_field, username)
        driver.execute_script("arguments[0].value = arguments[1];", password_field, password)
        
        print("✅ Credentials entered")
        
        # Try multiple submission methods
        print("Trying submission method 1: Enter key...")
        password_field.send_keys(Keys.RETURN)
        time.sleep(8)
        
        current_url = driver.current_url
        print(f"After Enter key: {current_url}")
        
        if "login.jsp" not in current_url:
            print("🎉 LOGIN SUCCESS!")
            print(f"Redirected to: {current_url}")
            
            # Look for download/data links
            try:
                data_links = driver.find_elements(By.XPATH, "//a[contains(text(),'Download') or contains(text(),'Data') or contains(text(),'Study')]")
                print(f"\nFound {len(data_links)} potential data links:")
                for link in data_links[:5]:
                    print(f"  - {link.text}: {link.get_attribute('href')}")
            except:
                pass
                
        else:
            print("❌ Still on login page")
            
            # Check for error messages
            try:
                errors = driver.find_elements(By.XPATH, "//*[contains(text(),'error') or contains(text(),'invalid') or contains(text(),'incorrect')]")
                if errors:
                    print("Error messages found:")
                    for error in errors:
                        if error.is_displayed():
                            print(f"  ⚠️ {error.text}")
            except:
                pass
        
        input("\nPress Enter to close browser...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()