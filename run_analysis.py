#!/usr/bin/env python3
"""
Agentic Alzheimer's Analyzer - Main Entry Point
===============================================

Launch autonomous analysis of Alzheimer's datasets using AI agents.
Designed for the ECOG-MemTrax experiment but generalizable to other datasets.
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the orchestrator
from core.orchestrator import AgenticAlzheimerAnalyzer

def main():
    """Main entry point"""
    print("""
    🧠 AGENTIC ALZHEIMER'S ANALYZER
    ===============================
    
    Autonomous AI agents for Alzheimer's research acceleration
    
    Features:
    • Automatic dataset discovery and characterization
    • ECOG-MemTrax correlation analysis 
    • Literature research and validation
    • AI-powered insights and recommendations
    • Grant-ready reports and visualizations
    • Token usage monitoring and control
    
    Starting analysis...
    """)
    
    try:
        # Initialize and run orchestrator
        analyzer = AgenticAlzheimerAnalyzer()
        results = analyzer.run_complete_analysis()
        
        if results['orchestrator']['status'] == 'completed':
            print("\n🎉 SUCCESS! Analysis completed successfully.")
            print("📁 Check the 'outputs/' directory for results.")
            print("🎯 Grant application materials are ready!")
            return 0
        else:
            print(f"\n❌ Analysis failed: {results['orchestrator'].get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)