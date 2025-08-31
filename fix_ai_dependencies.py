#!/usr/bin/env python3
"""
Fix AI Dependencies
==================
Script to resolve common AI provider dependency conflicts
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_dependencies():
    """Fix common AI dependency issues"""
    
    print("🔧 FIXING AI PROVIDER DEPENDENCIES")
    print("=" * 50)
    
    fixes_applied = []
    
    # Fix 1: OpenAI httpx conflict
    print("\n1. Fixing OpenAI httpx conflict...")
    success, stdout, stderr = run_command("pip install 'httpx<0.28.0'")
    if success:
        print("   ✅ httpx downgraded to compatible version")
        fixes_applied.append("httpx version fixed")
    else:
        print(f"   ❌ Failed to fix httpx: {stderr}")
    
    # Fix 2: Ensure compatible OpenAI version
    print("\n2. Installing compatible OpenAI version...")
    success, stdout, stderr = run_command("pip install 'openai>=1.0,<1.51'")
    if success:
        print("   ✅ OpenAI version set to compatible range")
        fixes_applied.append("OpenAI version fixed")
    else:
        print(f"   ❌ Failed to fix OpenAI: {stderr}")
    
    # Fix 3: Reinstall anthropic to ensure latest
    print("\n3. Ensuring latest Anthropic client...")
    success, stdout, stderr = run_command("pip install --upgrade anthropic")
    if success:
        print("   ✅ Anthropic client updated")
        fixes_applied.append("Anthropic updated")
    else:
        print(f"   ❌ Failed to update Anthropic: {stderr}")
    
    # Fix 4: Fix Gemini dependencies
    print("\n4. Fixing Gemini dependencies...")
    success, stdout, stderr = run_command("pip install --upgrade google-generativeai")
    if success:
        print("   ✅ Gemini dependencies updated")
        fixes_applied.append("Gemini dependencies fixed")
    else:
        print(f"   ❌ Failed to update Gemini: {stderr}")
    
    # Test installations
    print("\n🧪 TESTING AI PROVIDERS")
    print("=" * 30)
    
    # Test OpenAI
    try:
        import openai
        client = openai.OpenAI(api_key="test-key")
        print("   ✅ OpenAI client creation works")
    except Exception as e:
        print(f"   ❌ OpenAI still broken: {e}")
    
    # Test Anthropic
    try:
        import anthropic
        client = anthropic.Anthropic(api_key="test-key")
        print("   ✅ Anthropic client creation works")
    except Exception as e:
        print(f"   ❌ Anthropic issue: {e}")
    
    # Test Gemini
    try:
        import google.generativeai as genai
        print("   ✅ Gemini import works")
    except Exception as e:
        print(f"   ❌ Gemini issue: {e}")
    
    print(f"\n📋 SUMMARY")
    print("=" * 20)
    if fixes_applied:
        print("✅ Fixes applied:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        print("\n🚀 Try running the analysis again!")
    else:
        print("❌ No fixes could be applied automatically")
        print("💡 Manual intervention may be required")
    
    print("\n🔑 API KEY REMINDERS:")
    print("   export ANTHROPIC_API_KEY='your_key_here'")
    print("   export OPENAI_API_KEY='your_key_here'") 
    print("   export GEMINI_API_KEY='your_key_here'")

if __name__ == "__main__":
    fix_dependencies()