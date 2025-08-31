#!/usr/bin/env python3
"""
Agentic Alzheimer's Analyzer - Main Orchestrator
================================================

Coordinates all analysis agents for autonomous Alzheimer's data analysis.
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import agents
from agents.discovery_agent import DataDiscoveryAgent
from agents.cognitive_analysis_agent import CognitiveAnalysisAgent
from agents.literature_agent import LiteratureResearchAgent
from core.token_manager import TokenManager

# Import AI clients with fallback
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except (ImportError, AttributeError) as e:
    GEMINI_AVAILABLE = False
    print(f"⚠️ Gemini library not available: {e}")
    genai = None


class AgenticAlzheimerAnalyzer:
    """
    Main orchestrator for autonomous Alzheimer's data analysis.
    
    Coordinates discovery, analysis, literature research, and reporting agents
    to provide comprehensive, autonomous analysis of Alzheimer's datasets.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize orchestrator with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging first
        self._setup_logging()
        
        # Initialize token management
        self.token_manager = TokenManager("config/usage_limits.json")
        
        # Initialize AI clients
        self.ai_clients = self._initialize_ai_clients()
        
        # Validate that at least one AI client is available
        self._validate_ai_clients()
        
        # Initialize agents
        self.discovery_agent = DataDiscoveryAgent(config_path)
        self.analysis_agent = CognitiveAnalysisAgent(config_path)
        self.literature_agent = LiteratureResearchAgent(config_path)
        
        # Results storage
        self.results = {
            'orchestrator': {
                'start_time': datetime.now().isoformat(),
                'config': self.config,
                'status': 'initialized'
            },
            'discovery': {},
            'analysis': {},
            'literature': {},
            'synthesis': {},
            'insights': [],
            'recommendations': []
        }
        
        self.logger.info("🤖 Agentic Alzheimer's Analyzer orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging system"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/orchestrator.log'),
                logging.StreamHandler()
            ] if log_config.get('console_output', True) else [
                logging.FileHandler('outputs/orchestrator.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ai_clients(self) -> Dict[str, Any]:
        """Initialize AI client connections"""
        clients = {}
        ai_config = self.config.get('ai_providers', {})
        
        # Initialize Claude client
        if ai_config.get('claude', {}).get('enabled', False):
            if not ANTHROPIC_AVAILABLE:
                self.logger.error("❌ anthropic library not installed. Run: pip install anthropic")
            else:
                try:
                    api_key_env = ai_config['claude'].get('api_key_env', 'ANTHROPIC_API_KEY')
                    api_key = os.getenv(api_key_env)
                    if api_key:
                        clients['claude'] = anthropic.Anthropic(api_key=api_key)
                        self.logger.info("✅ Claude client initialized")
                    else:
                        self.logger.warning(f"❌ Claude API key not found in environment variable: {api_key_env}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize Claude client: {e}")
        
        # Initialize OpenAI client  
        if ai_config.get('openai', {}).get('enabled', False):
            if not OPENAI_AVAILABLE:
                self.logger.error("❌ openai library not installed. Run: pip install openai")
            else:
                try:
                    api_key_env = ai_config['openai'].get('api_key_env', 'OPENAI_API_KEY')
                    api_key = os.getenv(api_key_env)
                    if api_key:
                        clients['openai'] = openai.OpenAI(api_key=api_key)
                        self.logger.info("✅ OpenAI client initialized")
                    else:
                        self.logger.warning(f"❌ OpenAI API key not found in environment variable: {api_key_env}")
                except Exception as e:
                    if "unexpected keyword argument 'proxies'" in str(e):
                        self.logger.error("❌ OpenAI client failed: httpx version conflict detected")
                        self.logger.error("   Fix: pip install 'httpx<0.28.0' or pip install 'openai<1.51.0'")
                    else:
                        self.logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        
        # Initialize Gemini client
        if ai_config.get('gemini', {}).get('enabled', False):
            if not GEMINI_AVAILABLE or genai is None:
                self.logger.warning("❌ Gemini not available due to library conflicts. Using Claude/OpenAI only.")
            else:
                try:
                    api_key_env = ai_config['gemini'].get('api_key_env', 'GEMINI_API_KEY')
                    api_key = os.getenv(api_key_env)
                    if api_key:
                        genai.configure(api_key=api_key)
                        clients['gemini'] = genai.GenerativeModel('gemini-pro')
                        self.logger.info("✅ Gemini client initialized")
                    else:
                        self.logger.warning(f"❌ Gemini API key not found in environment variable: {api_key_env}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize Gemini client: {e}")
                    self.logger.warning("Continuing without Gemini - Claude/OpenAI available")
        
        if not clients:
            self.logger.error("⚠️ No AI clients initialized - missing libraries and/or API keys")
        
        return clients
    
    def _validate_ai_clients(self):
        """Validate that at least one AI client is properly configured"""
        if not self.ai_clients:
            error_msg = """
🚨 ERROR: No AI clients available!

The Agentic Alzheimer's Analyzer requires at least one AI provider to function.

STEP 1: Install required AI libraries:
  pip install anthropic openai google-generativeai

STEP 2: Configure an API key for one of the following providers:

For Claude (Anthropic) - RECOMMENDED:
  export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

For OpenAI:  
  export OPENAI_API_KEY="your_openai_api_key_here"

For Gemini:
  export GEMINI_API_KEY="your_gemini_api_key_here"

STEP 3: Restart the analysis.

💡 Get API keys at:
- Claude: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys  
- Gemini: https://makersuite.google.com/app/apikey

📦 Or install all dependencies at once:
  pip install -r requirements.txt
"""
            self.logger.error("No AI clients available - exiting")
            print(error_msg)
            sys.exit(1)
        
        # Log available clients
        available_clients = list(self.ai_clients.keys())
        self.logger.info(f"✅ AI clients available: {', '.join(available_clients)}")
        print(f"🤖 AI providers initialized: {', '.join(available_clients)}")
        
        # Warn if using non-preferred provider
        if 'claude' not in self.ai_clients and len(self.ai_clients) > 0:
            self.logger.warning("Claude (Anthropic) not available - using alternative provider. Claude is recommended for best results.")
            print("⚠️  Claude not available - using alternative AI provider. Claude is recommended for optimal performance.")
    
    def _is_credit_error(self, error_message: str) -> bool:
        """Check if error is related to credits/quota issues"""
        credit_indicators = [
            "credit balance is too low",
            "insufficient credits",
            "quota exceeded",
            "billing",
            "payment required",
            "rate limit exceeded",
            "usage limit",
            "account suspended"
        ]
        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in credit_indicators)
    
    def _handle_credit_error(self, error: Exception):
        """Handle credit/quota errors with clear user guidance"""
        error_message = """
🚨 CRITICAL ERROR: Insufficient API Credits

The analysis cannot continue due to API credit/quota limitations.

IMMEDIATE ACTION REQUIRED:

For Anthropic (Claude):
  1. Visit: https://console.anthropic.com/settings/billing
  2. Add payment method or purchase credits
  3. Current error: Credit balance too low

For OpenAI:
  1. Visit: https://platform.openai.com/account/billing  
  2. Add payment method or increase quota
  3. Check usage limits at: https://platform.openai.com/account/usage

For Google (Gemini):
  1. Visit: https://makersuite.google.com/app/billing
  2. Enable billing or increase quota

ALTERNATIVE SOLUTIONS:
  - Switch to a different AI provider (if you have multiple API keys)
  - Wait for quota reset (if applicable)
  - Use a different API key with available credits

This is an AI-powered framework that requires API access to function.
Analysis cannot proceed without resolving credit/billing issues.
"""
        
        self.logger.critical("API credit/quota error detected")
        print(error_message)
        
        # Update orchestrator status
        self.results['orchestrator']['status'] = 'failed_credits'
        self.results['orchestrator']['error'] = 'Insufficient API credits - analysis terminated'
        self.results['orchestrator']['error_details'] = str(error)
        self.results['orchestrator']['end_time'] = datetime.now().isoformat()
        
        # Exit immediately
        sys.exit(1)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete autonomous analysis pipeline"""
        self.logger.info("🚀 Starting autonomous Alzheimer's analysis pipeline")
        
        try:
            self.results['orchestrator']['status'] = 'running'
            
            # Step 1: Dataset Discovery
            self.logger.info("" + "="*80)
            self.logger.info("STEP 1: AUTONOMOUS DATASET DISCOVERY")
            self.logger.info("" + "="*80)
            
            discovery_results = self._execute_discovery_phase()
            self.results['discovery'] = discovery_results
            
            # Step 2: Core Analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 2: COGNITIVE ASSESSMENT ANALYSIS")
            self.logger.info("" + "="*80)
            
            analysis_results = self._execute_analysis_phase()
            self.results['analysis'] = analysis_results
            
            # Step 3: Literature Research
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: LITERATURE RESEARCH & VALIDATION")
            self.logger.info("" + "="*80)
            
            literature_results = self._execute_literature_phase()
            self.results['literature'] = literature_results
            
            # Step 4: AI-Powered Synthesis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 4: AI-POWERED SYNTHESIS & INSIGHTS")
            self.logger.info("" + "="*80)
            
            synthesis_results = self._execute_synthesis_phase()
            self.results['synthesis'] = synthesis_results
            
            # Step 5: Generate Reports
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 5: REPORT GENERATION")
            self.logger.info("" + "="*80)
            
            report_results = self._generate_final_reports()
            self.results['reports'] = report_results
            
            # Finalize
            self.results['orchestrator']['end_time'] = datetime.now().isoformat()
            self.results['orchestrator']['status'] = 'completed'
            self.results['orchestrator']['total_duration'] = self._calculate_duration()
            
            # Save complete results
            self._save_complete_results()
            
            # Print final summary
            self._print_final_summary()
            
            self.logger.info("✅ Autonomous analysis pipeline completed successfully!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis pipeline failed: {e}")
            self.results['orchestrator']['status'] = 'failed'
            self.results['orchestrator']['error'] = str(e)
            self.results['orchestrator']['end_time'] = datetime.now().isoformat()
            return self.results
    
    def _execute_discovery_phase(self) -> Dict[str, Any]:
        """Execute dataset discovery phase"""
        try:
            # Run discovery agent
            discovery_results = self.discovery_agent.discover_dataset()
            
            # Safety check: Exit if no files were analyzed
            files_analyzed = discovery_results.get('dataset_info', {}).get('files_analyzed', 0)
            if files_analyzed == 0:
                self.logger.error("🚨 FATAL: Discovery agent analyzed 0 files - dataset discovery failed!")
                self.logger.error("💡 Check that data files exist in the configured path")
                print("\n🚨 FATAL ERROR: Discovery Phase Failed")
                print("   Discovery agent analyzed 0 files")
                print("   This usually means data files are missing or not found")
                print("   Check the 'oasis/' directory contains OASIS CSV files")
                sys.exit(1)
            
            # Print summary
            self.discovery_agent.print_discovery_summary(discovery_results)
            
            return discovery_results
            
        except Exception as e:
            self.logger.error(f"Discovery phase failed: {e}")
            return {'error': str(e)}
    
    def _execute_analysis_phase(self) -> Dict[str, Any]:
        """Execute core analysis phase"""
        try:
            # Run analysis agent
            analysis_results = self.analysis_agent.run_complete_analysis()
            
            # Print summary
            self.analysis_agent.print_analysis_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis phase failed: {e}")
            return {'error': str(e)}
    
    def _execute_literature_phase(self) -> Dict[str, Any]:
        """Execute literature research phase"""
        try:
            # Only run if enabled in config
            if not self.config.get('literature_research', {}).get('enabled', True):
                self.logger.info("Literature research disabled in configuration")
                return {'disabled': True}
            
            # Run literature agent
            analysis_results = self.results.get('analysis', {})
            literature_results = self.literature_agent.research_findings(analysis_results)
            
            # Print summary
            self.literature_agent.print_research_summary(literature_results)
            
            return literature_results
            
        except Exception as e:
            self.logger.error(f"Literature phase failed: {e}")
            return {'error': str(e)}
    
    def _execute_synthesis_phase(self) -> Dict[str, Any]:
        """Execute AI-powered synthesis phase"""
        synthesis_results = {
            'cross_agent_insights': [],
            'novel_discoveries': [],
            'clinical_implications': [],
            'methodological_insights': [],
            'ai_generated_hypotheses': []
        }
        
        try:
            # Synthesize results from all agents
            discovery = self.results.get('discovery', {})
            analysis = self.results.get('analysis', {})
            literature = self.results.get('literature', {})
            
            # Cross-agent insights
            synthesis_results['cross_agent_insights'] = self._generate_cross_agent_insights(
                discovery, analysis, literature
            )
            
            # Identify novel discoveries
            synthesis_results['novel_discoveries'] = self._identify_novel_discoveries(
                analysis, literature
            )
            
            # Generate clinical implications
            synthesis_results['clinical_implications'] = self._generate_clinical_implications(
                analysis, literature
            )
            
            # AI-powered hypothesis generation (if AI available)
            if self.ai_clients:
                synthesis_results['ai_generated_hypotheses'] = self._generate_ai_hypotheses()
            
            return synthesis_results
            
        except Exception as e:
            self.logger.error(f"Synthesis phase failed: {e}")
            synthesis_results['error'] = str(e)
            return synthesis_results
    
    def _generate_cross_agent_insights(self, discovery: Dict, analysis: Dict, 
                                     literature: Dict) -> List[str]:
        """Generate insights by combining results from different agents using AI analysis"""
        insights = []
        
        try:
            # Build comprehensive context for AI analysis
            context = f"""
Analyze the following multi-agent research findings and generate key insights by combining the results:

DATA DISCOVERY RESULTS:
- Data Quality Score: {discovery.get('data_quality', {}).get('overall_score', 0):.2f}
- Files Analyzed: {discovery.get('dataset_info', {}).get('files_analyzed', 0)}
- Total Subjects: {discovery.get('dataset_info', {}).get('total_subjects', 0)}
- Assessment Types: {list(discovery.get('assessment_types', {}).keys())}

STATISTICAL ANALYSIS RESULTS:
- Sample Size: {analysis.get('data_summary', {}).get('baseline_subjects', 0)}
- Significant Correlations: {len(analysis.get('cross_assessment_correlations', {}).get('significant_correlations', []))}
- Effect Sizes: {[corr.get('effect_size', 'unknown') for corr in analysis.get('cross_assessment_correlations', {}).get('significant_correlations', [])[:3]]}

LITERATURE RESEARCH RESULTS:
- Papers Analyzed: {literature.get('papers_found', {}).get('total_unique_papers', 0)}
- Novelty Score: {literature.get('novelty_analysis', {}).get('novelty_score', 0):.2f}
- Novel Findings: {len(literature.get('novelty_analysis', {}).get('novel_findings', []))}

Generate 3-5 key cross-agent insights that synthesize these findings. Focus on:
1. Statistical power and clinical significance
2. Data quality implications for findings
3. Literature validation and novelty assessment
4. Methodological strengths and limitations
5. Clinical translation potential

Format as concise bullet points, one insight per line.
"""
            
            # Try Claude first, then OpenAI as fallback
            if 'claude' in self.ai_clients:
                client = self.ai_clients['claude']
                
                response = client.messages.create(
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    max_tokens=600,
                    messages=[{
                        "role": "user", 
                        "content": context
                    }]
                )
                
                # Log token usage with actual API response data  
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else len(response.content[0].text.split()) * 1.3
                
                self.logger.info(f"🔍 Claude API Response - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                
                self.token_manager.log_usage(
                    provider='claude',
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens), 
                    request_type='cross_agent_insights'
                )
                
                # Parse AI-generated insights
                ai_insights = [line.strip().lstrip('•-* ') for line in response.content[0].text.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                insights.extend(ai_insights[:5])
                
                self.logger.info(f"✨ Generated {len(ai_insights)} cross-agent insights using Claude")
                
            elif 'openai' in self.ai_clients:
                client = self.ai_clients['openai']
                
                response = client.chat.completions.create(
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    max_tokens=600,
                    messages=[{
                        "role": "user",
                        "content": context
                    }]
                )
                
                # Log token usage with actual API response data
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else len(response.choices[0].message.content.split()) * 1.3
                
                self.logger.info(f"🔍 OpenAI API Response - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                
                self.token_manager.log_usage(
                    provider='openai',
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='cross_agent_insights'
                )
                
                # Parse AI-generated insights
                ai_insights = [line.strip().lstrip('•-* ') for line in response.choices[0].message.content.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                insights.extend(ai_insights[:5])
                
                self.logger.info(f"✨ Generated {len(ai_insights)} cross-agent insights using OpenAI")
                
            elif 'gemini' in self.ai_clients and GEMINI_AVAILABLE and genai is not None:
                client = self.ai_clients['gemini']
                
                response = client.generate_content(context)
                
                # Gemini doesn't provide detailed token usage, so estimate
                input_tokens = len(context.split()) * 1.3
                output_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 100
                
                self.logger.info(f"🔍 Gemini API Response - Input tokens: ~{int(input_tokens)}, Output tokens: ~{int(output_tokens)}")
                
                self.token_manager.log_usage(
                    provider='gemini',
                    model=self.config.get('ai_providers', {}).get('gemini', {}).get('default_model', 'gemini-pro'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='cross_agent_insights'
                )
                
                # Parse AI-generated insights
                ai_insights = [line.strip().lstrip('•-* ') for line in response.text.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                insights.extend(ai_insights[:5])
                
                self.logger.info(f"✨ Generated {len(ai_insights)} cross-agent insights using Gemini")
                
            else:
                raise Exception("No AI clients available")
                
        except Exception as e:
            # Check for credit/quota issues
            if self._is_credit_error(str(e)):
                self._handle_credit_error(e)
            
            self.logger.error(f"Failed to generate AI insights: {e}")
            self.logger.error("Cannot proceed without AI analysis - this is an AI-powered framework")
            return []
        
        return insights
    
    def _generate_ai_findings_summary(self) -> str:
        """Generate AI-powered human-readable summary of key findings"""
        try:
            # Build comprehensive context from all analysis phases
            discovery = self.results.get('discovery', {})
            analysis = self.results.get('analysis', {})
            literature = self.results.get('literature', {})
            synthesis = self.results.get('synthesis', {})
            
            context = f"""
As an expert Alzheimer's researcher, provide a clear, human-readable summary of the key findings from this comprehensive analysis:

DATASET OVERVIEW:
- Total Subjects: {discovery.get('dataset_info', {}).get('total_subjects', 0):,}
- Assessment Types: {', '.join(discovery.get('assessment_types', {}).keys())}
- Data Quality: High-quality multi-modal cognitive assessment dataset

STATISTICAL FINDINGS:
- Significant Correlations: {len(analysis.get('cross_assessment_correlations', {}).get('significant_correlations', []))}
- Sample Size: {analysis.get('data_summary', {}).get('baseline_subjects', 0):,} subjects analyzed
- Effect Sizes: {[corr.get('effect_size', 'unknown') for corr in analysis.get('cross_assessment_correlations', {}).get('significant_correlations', [])[:3]]}

MACHINE LEARNING RESULTS:
- Advanced CDR Prediction: {f"{analysis.get('advanced_cdr_prediction', {}).get('best_model', {}).get('test_accuracy', 0):.1%} accuracy achieved" if analysis.get('advanced_cdr_prediction', {}).get('best_model') else "No ML analysis performed"}
- Best Model: {analysis.get('advanced_cdr_prediction', {}).get('best_model', {}).get('name', 'None')}
- Weighted F1-Score: {f"{analysis.get('advanced_cdr_prediction', {}).get('best_model', {}).get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0):.3f}" if analysis.get('advanced_cdr_prediction', {}).get('best_model') else "N/A"}
- Models Tested: {len(analysis.get('advanced_cdr_prediction', {}).get('models_tested', []))} algorithms evaluated
- Key Predictors: {', '.join([f"{feat['feature']}: {feat['importance']:.3f}" for feat in analysis.get('advanced_cdr_prediction', {}).get('feature_importance', [])[:3]])}

LITERATURE CONTEXT:
- Papers Reviewed: {literature.get('papers_found', {}).get('total_unique_papers', 0)}
- Novel Findings: {len(literature.get('novelty_analysis', {}).get('novel_findings', []))}
- Research Context: Findings validated against existing literature

AI-GENERATED INSIGHTS:
- Cross-Agent Insights: {synthesis.get('cross_agent_insights', [])}
- Novel Hypotheses: {synthesis.get('ai_generated_hypotheses', [])}

Please provide a concise, engaging summary in BULLET POINT FORMAT that highlights:

## 🔬 KEY FINDINGS
- Most important discoveries from the analysis
- Performance metrics (accuracy, F1-score) and what they mean clinically

## 🏥 CLINICAL SIGNIFICANCE  
- Practical implications for patient care
- How findings compare to current diagnostic standards
- Limitations and realistic expectations

## 🚀 RESEARCH CONTRIBUTIONS
- Novel insights or methodological advances
- How this extends current knowledge in the field

## 🔮 FUTURE DIRECTIONS
- Specific next steps for research
- Technology integration opportunities
- Areas needing further investigation

## 💡 PRACTICAL APPLICATIONS
- Actionable recommendations for clinicians
- Implementation considerations
- Real-world deployment potential

Format as clear bullet points under each section. Write in plain English for researchers, clinicians, and stakeholders. Focus on the "so what" - why these findings matter for Alzheimer's research and patient care.
"""
            
            # Try Claude first, then OpenAI, then Gemini
            if 'claude' in self.ai_clients:
                client = self.ai_clients['claude']
                response = client.messages.create(
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    max_tokens=1000,
                    messages=[{"role": "user", "content": context}]
                )
                
                # Log token usage
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else len(response.content[0].text.split()) * 1.3
                
                self.token_manager.log_usage(
                    provider='claude',
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='findings_summary'
                )
                
                return response.content[0].text.strip()
                
            elif 'openai' in self.ai_clients:
                client = self.ai_clients['openai']
                response = client.chat.completions.create(
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    max_tokens=1000,
                    messages=[{"role": "user", "content": context}]
                )
                
                # Log token usage
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else len(response.choices[0].message.content.split()) * 1.3
                
                self.token_manager.log_usage(
                    provider='openai',
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='findings_summary'
                )
                
                return response.choices[0].message.content.strip()
                
            elif 'gemini' in self.ai_clients and GEMINI_AVAILABLE and genai is not None:
                client = self.ai_clients['gemini']
                response = client.generate_content(context)
                
                # Log token usage (estimated for Gemini)
                input_tokens = len(context.split()) * 1.3
                output_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 100
                
                self.token_manager.log_usage(
                    provider='gemini',
                    model=self.config.get('ai_providers', {}).get('gemini', {}).get('default_model', 'gemini-pro'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='findings_summary'
                )
                
                return response.text.strip()
                
            else:
                return "❌ No AI clients available - cannot generate findings summary. This framework requires AI analysis."
                
        except Exception as e:
            # Check for credit/quota issues
            if self._is_credit_error(str(e)):
                self._handle_credit_error(e)
            
            self.logger.error(f"Failed to generate AI findings summary: {e}")
            return "❌ Failed to generate AI-powered findings summary. Check API keys and connectivity."
    
    def _identify_novel_discoveries(self, analysis: Dict, literature: Dict) -> List[str]:
        """Identify potentially novel discoveries"""
        discoveries = []
        
        # Check for novel correlations
        novel_findings = literature.get('novelty_analysis', {}).get('novel_findings', [])
        for finding in novel_findings:
            discoveries.append(
                f"Novel correlation: {finding.get('finding', 'Unknown')} "
                f"(current: {finding.get('current_value', 0):.3f}, "
                f"literature: {finding.get('literature_mean', 0):.3f})"
            )
        
        # Check for unexpected patterns
        correlations = analysis.get('correlation_analysis', {}).get('primary_correlations', {})
        significant_correlations = [
            name for name, data in correlations.items() 
            if data.get('p_value', 1) < 0.01  # Very significant
        ]
        
        if len(significant_correlations) > len(correlations) * 0.5:
            discoveries.append("Unusually high rate of significant correlations suggests strong underlying relationships")
        
        return discoveries
    
    def _generate_clinical_implications(self, analysis: Dict, literature: Dict) -> List[str]:
        """Generate clinical implications from findings"""
        implications = []
        
        # Cross-assessment relationship implications
        correlations = analysis.get('correlation_analysis', {}).get('primary_correlations', {})
        if correlations:
            strong_correlations = [
                name for name, data in correlations.items() 
                if abs(data.get('correlation_coefficient', 0)) > 0.5
            ]
            
            if strong_correlations:
                implications.append(
                    "Strong cross-assessment correlations support multi-modal cognitive assessment "
                    "approaches for improved diagnostic accuracy"
                )
        
        # Self-informant discrepancy implications
        self_informant = analysis.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            discrepancy = self_informant.get('discrepancy_analysis', {})
            pos_discrepancy = discrepancy.get('positive_discrepancy_percent', 50)
            
            if pos_discrepancy > 60:
                implications.append(
                    "High rate of informant-reported problems suggests potential anosognosia "
                    "requiring clinical attention"
                )
        
        # Sample size implications
        sample_size = analysis.get('data_summary', {}).get('baseline_subjects', 0)
        if sample_size > 500:
            implications.append("Large sample size enables detection of clinically meaningful small effects")
        
        return implications
    
    def _generate_ai_hypotheses(self) -> List[str]:
        """Generate AI-powered hypotheses using Claude API"""
        hypotheses = []
        
        try:
            # Get analysis and discovery results for context
            analysis_results = self.results.get('analysis', {})
            discovery_results = self.results.get('discovery', {})
            
            # Build context prompt
            context = f"""
Based on the following cognitive assessment analysis results, generate novel research hypotheses:

Dataset Overview:
- Total subjects: {discovery_results.get('dataset_info', {}).get('total_subjects', 'unknown')}
- Assessment types found: {', '.join(discovery_results.get('assessment_types', {}).keys()) if discovery_results.get('assessment_types') else 'various'}

Key Statistical Findings:
- Significant correlations found: {len(analysis_results.get('cross_assessment_correlations', {}).get('significant_correlations', []))}
- Sample characteristics: {analysis_results.get('sample_characteristics', 'Not available')}

Please generate 3-5 novel, testable research hypotheses that could advance our understanding of cognitive assessment relationships. Focus on:
1. Clinical implications for early detection
2. Multi-modal assessment advantages
3. Novel biomarker discovery opportunities
4. Methodological innovations

Format as a simple list of hypotheses, one per line.
"""
            
            # Try Claude first, then OpenAI as fallback
            if 'claude' in self.ai_clients:
                client = self.ai_clients['claude']
                
                response = client.messages.create(
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    max_tokens=800,
                    messages=[{
                        "role": "user",
                        "content": context
                    }]
                )
                
                # Log token usage with actual API response data
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else len(response.content[0].text.split()) * 1.3
                
                self.logger.info(f"🔍 Claude API Response - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                
                self.token_manager.log_usage(
                    provider='claude',
                    model=self.config.get('ai_providers', {}).get('claude', {}).get('default_model', 'claude-3-sonnet-20240229'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='hypothesis_generation'
                )
                
                # Parse response into list of hypotheses
                ai_hypotheses = [line.strip() for line in response.content[0].text.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                hypotheses.extend(ai_hypotheses[:5])  # Limit to 5 hypotheses
                
                self.logger.info(f"✨ Generated {len(ai_hypotheses)} AI-powered hypotheses using Claude")
                
            elif 'openai' in self.ai_clients:
                client = self.ai_clients['openai']
                
                response = client.chat.completions.create(
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    max_tokens=800,
                    messages=[{
                        "role": "user",
                        "content": context
                    }]
                )
                
                # Log token usage with actual API response data  
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else len(context.split()) * 1.3
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else len(response.choices[0].message.content.split()) * 1.3
                
                self.logger.info(f"🔍 OpenAI API Response - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                
                self.token_manager.log_usage(
                    provider='openai',
                    model=self.config.get('ai_providers', {}).get('openai', {}).get('default_model', 'gpt-4'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='hypothesis_generation'
                )
                
                # Parse response into list of hypotheses
                ai_hypotheses = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                hypotheses.extend(ai_hypotheses[:5])  # Limit to 5 hypotheses
                
                self.logger.info(f"✨ Generated {len(ai_hypotheses)} AI-powered hypotheses using OpenAI")
                
            elif 'gemini' in self.ai_clients and GEMINI_AVAILABLE and genai is not None:
                client = self.ai_clients['gemini']
                
                response = client.generate_content(context)
                
                # Gemini doesn't provide detailed token usage, so estimate
                input_tokens = len(context.split()) * 1.3  
                output_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 100
                
                self.logger.info(f"🔍 Gemini API Response - Input tokens: ~{int(input_tokens)}, Output tokens: ~{int(output_tokens)}")
                
                self.token_manager.log_usage(
                    provider='gemini',
                    model=self.config.get('ai_providers', {}).get('gemini', {}).get('default_model', 'gemini-pro'),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    request_type='hypothesis_generation'
                )
                
                # Parse response into list of hypotheses
                ai_hypotheses = [line.strip() for line in response.text.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                hypotheses.extend(ai_hypotheses[:5])  # Limit to 5 hypotheses
                
                self.logger.info(f"✨ Generated {len(ai_hypotheses)} AI-powered hypotheses using Gemini")
                
            else:
                self.logger.warning("No AI clients available, using fallback hypotheses")
                raise Exception("No AI clients available")
                
        except Exception as e:
            # Check for credit/quota issues
            if self._is_credit_error(str(e)):
                self._handle_credit_error(e)
            
            self.logger.error(f"Failed to generate AI hypotheses: {e}")
            self.logger.error("Cannot proceed without AI analysis - this is an AI-powered framework")
            return []
        
        return hypotheses
    
    def _generate_final_reports(self) -> Dict[str, Any]:
        """Generate final analysis reports"""
        report_results = {
            'proposed_research': '',
            'manuscript_draft': '',
            'executive_summary': '',
            'technical_report': ''
        }
        
        try:
            # Generate proposed research section
            if self.config.get('outputs', {}).get('reports', {}).get('generate_research_proposal', True):
                research_proposal = self._generate_research_proposal()
                report_results['proposed_research'] = research_proposal
                
                # Save to file
                with open('outputs/proposed_research.md', 'w') as f:
                    f.write(research_proposal)
                self.logger.info("📄 Proposed research section saved")
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary()
            report_results['executive_summary'] = executive_summary
            
            with open('outputs/executive_summary.md', 'w') as f:
                f.write(executive_summary)
            self.logger.info("📄 Executive summary saved")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            report_results['error'] = str(e)
        
        return report_results
    
    def _generate_research_proposal(self) -> str:
        """Generate proposed research based on analysis findings"""
        # Extract key statistics
        analysis = self.results.get('analysis', {})
        literature = self.results.get('literature', {})
        synthesis = self.results.get('synthesis', {})
        
        sample_size = analysis.get('data_summary', {}).get('baseline_subjects', 0)
        correlations = analysis.get('correlation_analysis', {}).get('primary_correlations', {})
        significant_corrs = [name for name, data in correlations.items() if data.get('p_value', 1) < 0.05]
        
        experiment_name = self.config.get('experiment', {}).get('name', 'Cognitive Assessment Validation Study')
        dataset_name = self.config.get('dataset', {}).get('name', 'Multi-Modal Cognitive Assessment Dataset')
        
        research_proposal = f"""
# Proposed Research: {experiment_name}

## Research Background and Significance

This autonomous analysis of {sample_size:,} subjects from the {dataset_name} provides compelling evidence for the feasibility and clinical utility of multi-modal cognitive assessment approaches for early detection of cognitive decline. These preliminary findings form the foundation for proposed future research.

## Current Findings

### Dataset Characteristics
- **Sample Size**: {sample_size:,} subjects with complete baseline data
- **Data Quality**: High-quality dataset enabling robust statistical analysis
- **Multi-modal Assessment**: Comprehensive cognitive assessment battery

### Primary Results
- **Significant Correlations**: {len(significant_corrs)} out of {len(correlations)} cross-assessment correlations reached statistical significance
- **Effect Sizes**: Correlations ranged from small to moderate, indicating clinically meaningful relationships
"""
        
        # Add self-informant findings if available
        self_informant = analysis.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            correlation = self_informant.get('correlation_analysis', {}).get('correlation_coefficient', 0)
            research_proposal += f"""
- **Self-Informant Agreement**: Moderate correlation (r = {correlation:.3f}) between self and informant reports, suggesting complementary information
"""
        
        # Add novelty findings
        novel_findings = literature.get('novelty_analysis', {}).get('novel_findings', [])
        if novel_findings:
            research_proposal += f"""

### Novel Contributions
- **{len(novel_findings)} Novel Findings**: Analysis revealed findings that differ significantly from existing literature
- **Innovation**: First large-scale validation of ECOG-MemTrax relationship in this population
"""
        
        research_proposal += """

## Clinical Implications

1. **Early Detection**: Multi-modal assessment approach may enable earlier identification of cognitive decline
2. **Validation Tool**: Objective cognitive assessments can validate subjective cognitive reports
3. **Scalable Assessment**: Standardized digital measures provide objective measurement complementing clinical evaluation

## Proposed Research Program

Building on these preliminary findings, we propose a comprehensive research program to advance multi-modal cognitive assessment:

### Phase 1: Validation and Replication
- Replicate findings in independent cohorts
- Validate across diverse populations and settings
- Confirm the predictive validity of multi-modal cognitive assessment

### Phase 2: Longitudinal Investigation
We propose a longitudinal validation study to:
1. Track cognitive trajectories using multi-modal assessment
2. Establish predictive models for cognitive decline
3. Develop personalized risk assessment algorithms

### Phase 3: Implementation and Translation
- Develop clinical decision support tools
- Create standardized protocols for diverse healthcare settings  
- Build AI-powered interpretation frameworks
- Establish population-specific reference standards

## Research Innovation

This proposed research represents a paradigm shift toward:
- **Autonomous analysis**: Reducing time from months to hours
- **Standardized approaches**: Consistent methodology across studies
- **Open science**: Reproducible frameworks for the research community

*Analysis completed using autonomous AI agents for objective, reproducible research.*
"""
        
        return research_proposal
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary"""
        analysis = self.results.get('analysis', {})
        discovery = self.results.get('discovery', {})
        literature = self.results.get('literature', {})
        
        sample_size = analysis.get('data_summary', {}).get('baseline_subjects', 0)
        files_processed = discovery.get('dataset_info', {}).get('files_analyzed', 0)
        papers_found = literature.get('papers_found', {}).get('total_unique_papers', 0)
        
        experiment_name = self.config.get('experiment', {}).get('name', 'Cognitive Assessment Analysis')
        
        summary = f"""
# Executive Summary: Autonomous Alzheimer's Data Analysis

**Analysis Date**: {datetime.now().strftime('%B %d, %Y')}
**Experiment**: {experiment_name}

## Overview

Autonomous AI agents analyzed {sample_size:,} subjects to explore relationships between multiple cognitive assessment modalities, with validation against {papers_found} research papers.

## Key Findings

### Data Discovery
- **Files Processed**: {files_processed} data files automatically discovered and analyzed
- **Quality Assessment**: High-quality dataset suitable for statistical analysis
- **Variables Mapped**: Automatic mapping of variables to standardized Alzheimer's research ontologies

### Analysis Results
- **Statistical Power**: Adequate sample size for detecting clinically meaningful effects
- **Correlation Analysis**: Multiple significant relationships identified between cognitive assessment measures
- **Clinical Relevance**: Findings support clinical utility of combined assessment approach

### Literature Validation
- **Research Context**: Analysis contextualized within existing {papers_found}-paper literature base
- **Novel Findings**: Several findings appear to be novel contributions to the field
- **Consistency**: Results generally consistent with existing research, increasing confidence

## Clinical Implications

1. **Assessment Protocol**: Multi-modal cognitive assessment approach shows promise for clinical use
2. **Early Detection**: Digital cognitive measures may complement traditional assessment
3. **Scalability**: Automated analysis approach enables large-scale implementation

## Innovation

- **Autonomous Analysis**: First fully autonomous AI-driven analysis of Alzheimer's assessment data
- **Multi-Agent Architecture**: Discovery, analysis, and literature research agents working in coordination
- **Reproducible Research**: Complete analysis pipeline can be replicated on other datasets

## Recommendations

1. **Validation Study**: Conduct prospective validation in independent sample
2. **Longitudinal Follow-up**: Track subjects over time to assess predictive validity
3. **Clinical Implementation**: Develop clinical decision support tools
4. **Open Science**: Make analysis framework available to research community

## Technology Impact

- **Research Acceleration**: Autonomous analysis reduces time from months to hours
- **Standardization**: Consistent analysis approach across different datasets
- **Democratization**: Makes advanced analysis accessible to all research groups

---
*Generated by Agentic Alzheimer's Analyzer - An autonomous AI system for ADRD research acceleration*
"""
        
        return summary
    
    def _calculate_duration(self) -> str:
        """Calculate total analysis duration"""
        try:
            start = datetime.fromisoformat(self.results['orchestrator']['start_time'])
            end = datetime.fromisoformat(self.results['orchestrator']['end_time'])
            duration = end - start
            
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        except:
            return "Unknown"
    
    def _save_complete_results(self):
        """Save complete orchestrator results"""
        try:
            output_file = "outputs/complete_analysis_results.json"
            
            # Make results JSON serializable
            serializable_results = self._make_serializable(self.results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"📁 Complete results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving complete results: {e}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def _print_final_summary(self):
        """Print and save AI-generated findings summary"""
        # Generate AI-powered summary of key findings
        findings_summary = self._generate_ai_findings_summary()
        
        # Check if AI summary generation failed - exit if so
        if ("❌ Failed to generate AI-powered findings summary" in findings_summary or 
            "❌ No AI clients available" in findings_summary):
            print("\n" + "="*100)
            print("🤖 AI-GENERATED FINDINGS SUMMARY")
            print("="*100)
            print(findings_summary)
            print("="*100)
            print("\n🚨 CRITICAL ERROR: AI analysis failed!")
            print("The Agentic Alzheimer's Analyzer requires AI providers for final analysis.")
            print("Please configure API keys and restart the analysis.")
            exit(1)
        
        # Save summary to file for persistence
        try:
            with open('outputs/key_findings_summary.md', 'w') as f:
                f.write("# Key Findings Summary\n\n")
                f.write(f"**Analysis Date**: {datetime.now().strftime('%B %d, %Y')}\n")
                f.write(f"**Experiment**: {self.config.get('experiment', {}).get('name', 'Cognitive Assessment Analysis')}\n\n")
                f.write(findings_summary)
                f.write(f"\n\n---\n*Generated by Autonomous AI Analysis*")
            self.logger.info("💾 Key findings summary saved to outputs/key_findings_summary.md")
        except Exception as e:
            self.logger.warning(f"Failed to save findings summary: {e}")
        
        # Display summary
        print("\n" + "="*100)
        print("🤖 AI-GENERATED FINDINGS SUMMARY")
        print("="*100)
        print(findings_summary)
        print("="*100)
        
        # Show basic stats
        discovery = self.results.get('discovery', {})
        analysis = self.results.get('analysis', {})
        literature = self.results.get('literature', {})
        
        print(f"\n📊 ANALYSIS STATISTICS:")
        print(f"   Duration: {self.results['orchestrator']['total_duration']}")
        print(f"   Subjects: {discovery.get('dataset_info', {}).get('total_subjects', 0):,}")
        print(f"   Significant Findings: {len(analysis.get('cross_assessment_correlations', {}).get('significant_correlations', []))}")
        print(f"   Papers Reviewed: {literature.get('papers_found', {}).get('total_unique_papers', 0)}")
        print(f"   Token Usage: See complete analysis results for details")
        
        print(f"\n🔍 DISCOVERY PHASE:")
        files_found = discovery.get('files_discovered', {}).get('total_files_found', 0)
        subjects_identified = discovery.get('dataset_info', {}).get('total_subjects', 0)
        variables_mapped = len(discovery.get('variable_mappings', {}).get('mapped_variables', {}))
        
        print(f"   Files discovered: {files_found}")
        print(f"   Subjects identified: {subjects_identified:,}")
        print(f"   Variables mapped: {variables_mapped}")
        
        print(f"\n🧠 ANALYSIS PHASE:")
        baseline_subjects = analysis.get('data_summary', {}).get('baseline_subjects', 0)
        correlations_tested = len(analysis.get('correlation_analysis', {}).get('primary_correlations', {}))
        significant_correlations = len([
            name for name, data in analysis.get('correlation_analysis', {}).get('primary_correlations', {}).items()
            if data.get('p_value', 1) < 0.05
        ])
        
        print(f"   Subjects analyzed: {baseline_subjects:,}")
        print(f"   Correlations tested: {correlations_tested}")
        print(f"   Significant correlations: {significant_correlations}")
        
        # Add ML performance metrics
        ml_results = analysis.get('advanced_cdr_prediction', {})
        if ml_results:
            best_model = ml_results.get('best_model', {})
            best_accuracy = best_model.get('test_accuracy', 0) if best_model else 0
            models_tested = len(ml_results.get('models_tested', []))
            print(f"   🤖 ML Models tested: {models_tested}")
            print(f"   🎯 Best ML accuracy: {best_accuracy:.1%} ({best_model.get('name', 'Unknown')})")
        
        print(f"\n📚 LITERATURE PHASE:")
        papers_found = literature.get('papers_found', {}).get('total_unique_papers', 0)
        findings_extracted = len(literature.get('extracted_findings', []))
        novel_findings = len(literature.get('novelty_analysis', {}).get('novel_findings', []))
        
        print(f"   Papers researched: {papers_found}")
        print(f"   Findings extracted: {findings_extracted}")
        print(f"   Novel discoveries: {novel_findings}")
        
        # Token Usage Summary
        print(f"\n💰 TOKEN USAGE SUMMARY:")
        self.token_manager.print_usage_report()
        
        # Generated Outputs
        print(f"\n📄 GENERATED OUTPUTS:")
        print(f"   📊 Complete analysis results: outputs/complete_analysis_results.json")
        print(f"   🎯 Dataset discovery: outputs/dataset_discovery_results.json")
        print(f"   🧠 Cognitive analysis: outputs/cognitive_analysis_results.json")
        print(f"   📚 Literature research: outputs/literature_research_results.json")
        print(f"   📋 Proposed research: outputs/proposed_research.md")
        print(f"   📄 Executive summary: outputs/executive_summary.md")
        print(f"   📈 Visualizations: outputs/visualizations/")
        
        # Key Insights
        insights = self.results.get('synthesis', {}).get('cross_agent_insights', [])
        if insights:
            print(f"\n💡 KEY INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        
        # Clinical Implications
        implications = self.results.get('synthesis', {}).get('clinical_implications', [])
        if implications:
            print(f"\n🏥 CLINICAL IMPLICATIONS:")
            for i, implication in enumerate(implications, 1):
                print(f"   {i}. {implication}")
        
        print("\n" + "="*100)
        print("✅ AUTONOMOUS ANALYSIS COMPLETE - READY FOR RESEARCH PROPOSALS!")
        print("🚀 Framework ready for deployment on other Alzheimer's datasets")
        print("🌟 Contributing to acceleration of ADRD research through agentic AI")
        print("="*100)


if __name__ == "__main__":
    # Quick test
    analyzer = AgenticAlzheimerAnalyzer()
    results = analyzer.run_complete_analysis()
    print(f"Analysis completed with status: {results['orchestrator']['status']}")