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
import argparse

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the orchestrator
from core.orchestrator import AgenticAlzheimerAnalyzer


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Agentic Alzheimer's Analyzer")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override config.dataset.name (e.g., Generic_CSV_Kaggle)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (no external AI calls)")
    parser.add_argument("--data-path", type=str, default=None, help="Override dataset data path (file or directory)")
    parser.add_argument("--limit-rows", type=int, default=None, help="Enable sampling and set max rows per dataset")
    args = parser.parse_args()

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
        overrides = {}
        if args.dataset_name:
            overrides = {"dataset": {"name": args.dataset_name}}
        # Apply offline override
        if args.offline:
            overrides.setdefault('ai_settings', {})['offline_mode'] = True
        # Apply data path override
        if args.data_path:
            ds_entry = {
                'path': args.data_path,
                'type': 'local_directory' if os.path.isdir(args.data_path) else 'local_file',
                'description': 'CLI-provided data source'
            }
            overrides.setdefault('dataset', {}).setdefault('data_sources', [])
            overrides['dataset']['data_sources'] = [ds_entry]
        # Apply sampling override
        if args.limit_rows and args.limit_rows > 0:
            overrides.setdefault('analysis', {})['use_sampling'] = True
            overrides.setdefault('analysis', {})['analysis_sample_size'] = int(args.limit_rows)
        
        # Initialize and run orchestrator
        analyzer = AgenticAlzheimerAnalyzer(config_path=args.config, overrides=overrides)
        results = analyzer.run_complete_analysis()
        
        if results['orchestrator']['status'] == 'completed':
            print("\n🎉 SUCCESS! Analysis completed successfully.")
            print("📁 Check the 'outputs/' directory for results.")
            print("🎯 Research proposal materials are ready!")
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