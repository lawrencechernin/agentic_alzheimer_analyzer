#!/usr/bin/env python3
"""
Literature Research Agent
=========================

Autonomous agent for researching existing literature to validate findings
and identify novel discoveries. Integrates with PubMed and other sources.
"""

import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from urllib.parse import quote
import re
from collections import defaultdict

class LiteratureResearchAgent:
    """
    Autonomous literature research and validation agent
    
    Capabilities:
    - PubMed/NCBI literature search
    - Semantic Scholar integration
    - Automated citation extraction
    - Effect size comparison
    - Novelty detection
    - Literature synthesis
    - AI-powered analysis of findings
    """
    
    def __init__(self, config_path: str = "config/config.yaml",
                 ai_client=None, token_manager=None):
        self.config_path = config_path
        self.ai_client = ai_client
        self.token_manager = token_manager
        
        # Setup logging early
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration (graceful if missing)
        self.config = self._load_config()
        self.literature_config = self.config.get('literature_research', {})
        
        # API endpoints
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1/"
        
        # Results storage
        self.search_results = {}
        self.extracted_findings = []
        self.novelty_analysis = {}
        
        self.logger.info("📚 Literature Research Agent initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            import yaml
            cfg_path = self.config_path
            if not os.path.exists(cfg_path):
                self.logger.warning(f"Warning: Could not load config {cfg_path}: file not found. Using defaults.")
                default_path = "config/config.yaml"
                if os.path.exists(default_path):
                    cfg_path = default_path
                else:
                    return {}
            with open(cfg_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def research_findings(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Research literature to validate and contextualize findings"""
        self.logger.info("🔍 Starting literature research and validation")
        
        research_results = {
            'research_timestamp': datetime.now().isoformat(),
            'search_queries': [],
            'papers_found': {},
            'extracted_findings': [],
            'novelty_analysis': {},
            'literature_synthesis': {},
            'validation_results': {}
        }
        
        try:
            # Step 1: Generate search queries from analysis results
            self.logger.info("Step 1: Generating search queries")
            search_queries = self._generate_search_queries(analysis_results)
            research_results['search_queries'] = search_queries
            
            # Step 2: Search literature databases
            self.logger.info("Step 2: Searching literature databases")
            papers_found = self._search_literature_databases(search_queries)
            research_results['papers_found'] = papers_found
            
            # Step 3: Extract relevant findings
            self.logger.info("Step 3: Extracting relevant findings")
            extracted_findings = self._extract_relevant_findings(papers_found)
            research_results['extracted_findings'] = extracted_findings
            
            # Step 4: Analyze novelty of current findings
            self.logger.info("Step 4: Analyzing novelty")
            novelty_analysis = self._analyze_finding_novelty(analysis_results, extracted_findings)
            research_results['novelty_analysis'] = novelty_analysis
            
            # Step 5: Synthesize literature
            self.logger.info("Step 5: Synthesizing literature")
            literature_synthesis = self._synthesize_literature(extracted_findings)
            research_results['literature_synthesis'] = literature_synthesis
            
            # Step 6: Validate findings against literature
            self.logger.info("Step 6: Validating findings")
            validation_results = self._validate_findings(analysis_results, extracted_findings)
            research_results['validation_results'] = validation_results
            
            # Save results
            self._save_research_results(research_results)
            
            self.logger.info("✅ Literature research complete!")
            return research_results
            
        except Exception as e:
            self.logger.error(f"Literature research failed: {e}")
            research_results['error'] = str(e)
            return research_results
    
    def _generate_search_queries(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate targeted search queries based on analysis results"""
        queries = []
        
        # Base queries from configuration
        base_queries = self.literature_config.get('search_terms', [])
        queries.extend(base_queries)
        
        # Generate queries based on specific findings
        correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        
        if correlations:
            # Add ECOG-specific queries
            queries.extend([
                "ECOG everyday cognition questionnaire correlation",
                "ECOG self-report informant discrepancy",
                "everyday cognition digital assessment correlation"
            ])
        
        # Add MemTrax-specific queries if MemTrax was used
        if any('memtrax' in str(key).lower() for key in correlations.keys()):
            queries.extend([
                "MemTrax cognitive assessment validation",
                "reaction time cognitive screening Alzheimer",
                "digital cognitive assessment reaction time"
            ])
        
        # Add self-awareness queries if relevant
        self_informant = analysis_results.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            queries.extend([
                "anosognosia Alzheimer self-awareness",
                "self-report informant agreement cognitive decline",
                "insight cognitive impairment Alzheimer"
            ])
        
        self.logger.info(f"   Generated {len(queries)} search queries")
        return queries[:15]  # Limit to prevent excessive API calls
    
    def _search_literature_databases(self, search_queries: List[str]) -> Dict[str, List[Dict]]:
        """Search multiple literature databases"""
        papers_found = {
            'pubmed': [],
            'semantic_scholar': [],
            'total_unique_papers': 0
        }
        
        max_papers_per_query = self.literature_config.get('max_papers_per_query', 20)
        
        # Search PubMed
        if 'pubmed' in self.literature_config.get('databases', []):
            self.logger.info("   🔍 Searching PubMed...")
            for query in search_queries[:5]:  # Limit queries to prevent overload
                try:
                    pubmed_results = self._search_pubmed(query, max_papers_per_query)
                    papers_found['pubmed'].extend(pubmed_results)
                    time.sleep(1)  # Respect API rate limits
                except Exception as e:
                    self.logger.warning(f"PubMed search failed for '{query}': {e}")
        
        # Search Semantic Scholar
        if 'semantic_scholar' in self.literature_config.get('databases', []):
            self.logger.info("   🔍 Searching Semantic Scholar...")
            for query in search_queries[:3]:  # Even more conservative
                try:
                    scholar_results = self._search_semantic_scholar(query, max_papers_per_query)
                    papers_found['semantic_scholar'].extend(scholar_results)
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
        
        # Remove duplicates and count
        all_papers = papers_found['pubmed'] + papers_found['semantic_scholar']
        unique_titles = set()
        unique_papers = []
        
        for paper in all_papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in unique_titles:
                unique_titles.add(title)
                unique_papers.append(paper)
        
        papers_found['total_unique_papers'] = len(unique_papers)
        
        self.logger.info(f"   📚 Found {len(unique_papers)} unique papers across databases")
        return papers_found
    
    def _search_pubmed(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search PubMed database"""
        results = []
        
        try:
            # Search for PMIDs
            search_url = f"{self.pubmed_base}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=30)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if pmids:
                # Fetch paper details
                fetch_url = f"{self.pubmed_base}efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'xml',
                    'rettype': 'abstract'
                }
                
                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
                fetch_response.raise_for_status()
                
                # Parse XML (simplified)
                xml_content = fetch_response.text
                papers = self._parse_pubmed_xml(xml_content, pmids)
                results.extend(papers)
            
        except Exception as e:
            self.logger.warning(f"PubMed search error: {e}")
        
        return results
    
    def _parse_pubmed_xml(self, xml_content: str, pmids: List[str]) -> List[Dict[str, Any]]:
        """Parse PubMed XML response (simplified)"""
        papers = []
        
        try:
            # Very basic XML parsing - in production, use proper XML parser
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic information
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    year_elem = article.find('.//PubDate/Year')
                    year = year_elem.text if year_elem is not None else "Unknown"
                    
                    # Get PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    papers.append({
                        'title': title,
                        'abstract': abstract[:1000],  # Truncate for storage
                        'year': year,
                        'pmid': pmid,
                        'source': 'pubmed',
                        'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}' if pmid else None
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing individual article: {e}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"XML parsing error: {e}")
            # Fallback - create basic entries for PMIDs
            for pmid in pmids[:5]:  # Limit fallback
                papers.append({
                    'title': f'PubMed Article {pmid}',
                    'abstract': '',
                    'year': 'Unknown',
                    'pmid': pmid,
                    'source': 'pubmed',
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}'
                })
        
        return papers
    
    def _search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search Semantic Scholar database"""
        results = []
        
        try:
            search_url = f"{self.semantic_scholar_base}paper/search"
            params = {
                'query': query,
                'limit': min(max_results, 50),  # API limit
                'fields': 'title,abstract,year,authors,citationCount,url'
            }
            
            headers = {'User-Agent': 'AlzheimerResearchBot/1.0'}
            
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                for paper in papers:
                    results.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'abstract': (paper.get('abstract') or '')[:1000],  # Truncate
                        'year': paper.get('year', 'Unknown'),
                        'authors': ', '.join([author.get('name', '') for author in paper.get('authors', [])[:3]]),
                        'citation_count': paper.get('citationCount', 0),
                        'source': 'semantic_scholar',
                        'url': paper.get('url')
                    })
            
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search error: {e}")
        
        return results
    
    def _extract_relevant_findings(self, papers_found: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Extract relevant findings from discovered papers"""
        extracted_findings = []
        
        all_papers = papers_found.get('pubmed', []) + papers_found.get('semantic_scholar', [])
        
        for paper in all_papers[:30]:  # Limit processing
            try:
                findings = self._extract_paper_findings(paper)
                if findings:
                    extracted_findings.extend(findings)
            except Exception as e:
                self.logger.warning(f"Error extracting findings from paper: {e}")
                continue
        
        self.logger.info(f"   📖 Extracted {len(extracted_findings)} findings from literature")
        return extracted_findings
    
    def _extract_paper_findings(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from a single paper"""
        findings = []
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Look for correlation coefficients
        correlation_patterns = [
            r'r\s*=\s*([0-9.-]+)',
            r'correlation\s*[^0-9]*([0-9.-]+)',
            r'correlated\s*\([^)]*r\s*=\s*([0-9.-]+)'
        ]
        
        for pattern in correlation_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            for match in matches:
                try:
                    correlation = float(match)
                    if -1 <= correlation <= 1:
                        findings.append({
                            'type': 'correlation',
                            'value': correlation,
                            'context': f"From paper: {title[:100]}...",
                            'source_paper': paper,
                            'extracted_text': f"Correlation: {match}"
                        })
                except ValueError:
                    continue
        
        # Look for effect sizes
        effect_size_patterns = [
            r"Cohen['\"]?s\s*d\s*=\s*([0-9.-]+)",
            r'effect\s*size[^0-9]*([0-9.-]+)',
            r'd\s*=\s*([0-9.-]+)'
        ]
        
        for pattern in effect_size_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            for match in matches:
                try:
                    effect_size = float(match)
                    if -3 <= effect_size <= 3:  # Reasonable range
                        findings.append({
                            'type': 'effect_size',
                            'value': effect_size,
                            'context': f"From paper: {title[:100]}...",
                            'source_paper': paper,
                            'extracted_text': f"Effect size: {match}"
                        })
                except ValueError:
                    continue
        
        # Look for sample sizes
        sample_patterns = [
            r'n\s*=\s*([0-9,]+)',
            r'([0-9,]+)\s*participants',
            r'([0-9,]+)\s*subjects'
        ]
        
        for pattern in sample_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            for match in matches[:2]:  # Limit to prevent overload
                try:
                    # Clean up number (remove commas)
                    sample_size = int(match.replace(',', ''))
                    if 10 <= sample_size <= 100000:  # Reasonable range
                        findings.append({
                            'type': 'sample_size',
                            'value': sample_size,
                            'context': f"From paper: {title[:100]}...",
                            'source_paper': paper,
                            'extracted_text': f"Sample size: {match}"
                        })
                except ValueError:
                    continue
        
        return findings
    
    def _analyze_finding_novelty(self, analysis_results: Dict[str, Any], 
                                extracted_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze novelty of current findings compared to literature"""
        novelty_analysis = {
            'novel_findings': [],
            'confirmatory_findings': [],
            'contradictory_findings': [],
            'novelty_score': 0.0
        }
        
        # Get current study correlations
        current_correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        
        # Extract literature correlations
        literature_correlations = [f for f in extracted_findings if f.get('type') == 'correlation']
        
        # Compare current findings to literature
        for corr_name, corr_data in current_correlations.items():
            current_r = corr_data.get('correlation_coefficient', 0)
            
            # Find similar correlations in literature
            similar_lit_correlations = []
            for lit_finding in literature_correlations:
                # Simple matching based on keywords (could be improved)
                if any(keyword in corr_name.lower() for keyword in ['ecog', 'memory', 'cognitive']):
                    similar_lit_correlations.append(lit_finding['value'])
            
            if similar_lit_correlations:
                import numpy as np
                lit_mean = np.mean(similar_lit_correlations)
                difference = abs(current_r - lit_mean)
                
                threshold = self.literature_config.get('novelty_thresholds', {}).get('correlation_difference', 0.1)
                
                if difference > threshold:
                    novelty_analysis['novel_findings'].append({
                        'finding': corr_name,
                        'current_value': current_r,
                        'literature_mean': lit_mean,
                        'difference': difference,
                        'interpretation': 'Novel finding - differs significantly from literature'
                    })
                else:
                    novelty_analysis['confirmatory_findings'].append({
                        'finding': corr_name,
                        'current_value': current_r,
                        'literature_mean': lit_mean,
                        'difference': difference,
                        'interpretation': 'Confirms existing literature'
                    })
        
        # Calculate novelty score
        total_findings = len(novelty_analysis['novel_findings']) + len(novelty_analysis['confirmatory_findings'])
        if total_findings > 0:
            novelty_analysis['novelty_score'] = len(novelty_analysis['novel_findings']) / total_findings
        
        self.logger.info(f"   🆕 Novelty analysis: {len(novelty_analysis['novel_findings'])} novel findings")
        
        return novelty_analysis
    
    def _synthesize_literature(self, extracted_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize literature findings"""
        synthesis = {
            'correlation_summary': {},
            'effect_size_summary': {},
            'sample_size_summary': {},
            'key_papers': [],
            'research_gaps': []
        }
        
        # Analyze correlations
        correlations = [f['value'] for f in extracted_findings if f.get('type') == 'correlation']
        if correlations:
            import numpy as np
            synthesis['correlation_summary'] = {
                'mean': float(np.mean(correlations)),
                'median': float(np.median(correlations)),
                'std': float(np.std(correlations)),
                'count': len(correlations),
                'range': [float(min(correlations)), float(max(correlations))]
            }
        
        # Analyze effect sizes
        effect_sizes = [f['value'] for f in extracted_findings if f.get('type') == 'effect_size']
        if effect_sizes:
            import numpy as np
            synthesis['effect_size_summary'] = {
                'mean': float(np.mean(effect_sizes)),
                'median': float(np.median(effect_sizes)),
                'std': float(np.std(effect_sizes)),
                'count': len(effect_sizes)
            }
        
        # Analyze sample sizes
        sample_sizes = [f['value'] for f in extracted_findings if f.get('type') == 'sample_size']
        if sample_sizes:
            import numpy as np
            synthesis['sample_size_summary'] = {
                'mean': float(np.mean(sample_sizes)),
                'median': float(np.median(sample_sizes)),
                'range': [int(min(sample_sizes)), int(max(sample_sizes))],
                'count': len(sample_sizes)
            }
        
        # Identify key papers (those with multiple findings)
        paper_counts = defaultdict(int)
        for finding in extracted_findings:
            paper = finding.get('source_paper', {})
            title = paper.get('title', 'Unknown')
            paper_counts[title] += 1
        
        # Get top papers by finding count
        top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        synthesis['key_papers'] = [{'title': title, 'findings_count': count} for title, count in top_papers]
        
        # Identify research gaps (simplified)
        synthesis['research_gaps'] = [
            "Limited longitudinal studies of ECOG-digital cognitive assessment relationships",
            "Need for larger sample sizes in validation studies",
            "Insufficient diversity in study populations"
        ]
        
        return synthesis
    
    def _validate_findings(self, analysis_results: Dict[str, Any], 
                          extracted_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate current findings against literature"""
        validation_results = {
            'literature_support': {},
            'consistency_analysis': {},
            'statistical_power_comparison': {},
            'recommendations': []
        }
        
        # Analyze literature support for key findings
        current_correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        literature_correlations = [f for f in extracted_findings if f.get('type') == 'correlation']
        
        if current_correlations and literature_correlations:
            validation_results['literature_support']['correlation_studies_found'] = len(literature_correlations)
            validation_results['literature_support']['current_studies_count'] = len(current_correlations)
            
            # Simple consistency check
            if len(literature_correlations) > 0:
                import numpy as np
                lit_mean = np.mean([f['value'] for f in literature_correlations])
                current_mean = np.mean([c.get('correlation_coefficient', 0) for c in current_correlations.values()])
                
                validation_results['consistency_analysis'] = {
                    'literature_mean_correlation': lit_mean,
                    'current_mean_correlation': current_mean,
                    'difference': abs(current_mean - lit_mean),
                    'consistent': abs(current_mean - lit_mean) < 0.2
                }
        
        # Sample size comparison
        current_sample_sizes = []
        data_summary = analysis_results.get('data_summary', {})
        if 'baseline_subjects' in data_summary:
            current_sample_sizes.append(data_summary['baseline_subjects'])
        
        literature_samples = [f['value'] for f in extracted_findings if f.get('type') == 'sample_size']
        
        if current_sample_sizes and literature_samples:
            import numpy as np
            validation_results['statistical_power_comparison'] = {
                'current_sample_size': max(current_sample_sizes),
                'literature_median_sample_size': int(np.median(literature_samples)),
                'percentile_rank': self._calculate_percentile_rank(max(current_sample_sizes), literature_samples)
            }
        
        # Generate recommendations
        if validation_results.get('consistency_analysis', {}).get('consistent', False):
            validation_results['recommendations'].append("Findings consistent with existing literature")
        else:
            validation_results['recommendations'].append("Findings differ from literature - investigate further")
        
        if current_sample_sizes and literature_samples:
            current_size = max(current_sample_sizes)
            median_lit_size = int(np.median(literature_samples))
            if current_size < median_lit_size:
                validation_results['recommendations'].append("Consider increasing sample size for stronger evidence")
        
        return validation_results
    
    def _calculate_percentile_rank(self, value: float, reference_list: List[float]) -> float:
        """Calculate percentile rank of value in reference list"""
        if not reference_list:
            return 0.0
        
        count_below = sum(1 for x in reference_list if x < value)
        return count_below / len(reference_list) * 100
    
    def _save_research_results(self, results: Dict[str, Any]):
        """Save literature research results"""
        output_file = "outputs/literature_research_results.json"
        
        try:
            os.makedirs("outputs", exist_ok=True)
            
            # Make results JSON serializable
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"📁 Literature research results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__class__') and 'numpy' in str(obj.__class__):
            # Handle numpy types more robustly
            if hasattr(obj, 'item'):
                return obj.item()
            else:
                return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif str(type(obj)) in ['<class \'pandas._libs.missing.NAType\'>', '<class \'numpy.float64\'>', '<class \'numpy.int64\'>']:
            return None if 'NAType' in str(type(obj)) else (float(obj) if 'float' in str(type(obj)) else int(obj))
        else:
            return obj
    
    def print_research_summary(self, results: Dict[str, Any]):
        """Print formatted literature research summary"""
        print("\n" + "="*80)
        print("📚 LITERATURE RESEARCH SUMMARY")
        print("="*80)
        
        # Search overview
        papers_found = results.get('papers_found', {})
        print(f"\n🔍 SEARCH OVERVIEW:")
        print(f"   Search queries: {len(results.get('search_queries', []))}")
        print(f"   Papers found: {papers_found.get('total_unique_papers', 0)}")
        print(f"   PubMed papers: {len(papers_found.get('pubmed', []))}")
        print(f"   Semantic Scholar papers: {len(papers_found.get('semantic_scholar', []))}")
        
        # Findings extraction
        extracted_findings = results.get('extracted_findings', [])
        if extracted_findings:
            print(f"\n📖 EXTRACTED FINDINGS:")
            print(f"   Total findings extracted: {len(extracted_findings)}")
            
            finding_types = defaultdict(int)
            for finding in extracted_findings:
                finding_types[finding.get('type', 'unknown')] += 1
            
            for finding_type, count in finding_types.items():
                print(f"   {finding_type}: {count}")
        
        # Novelty analysis
        novelty = results.get('novelty_analysis', {})
        if novelty:
            print(f"\n🆕 NOVELTY ANALYSIS:")
            print(f"   Novel findings: {len(novelty.get('novel_findings', []))}")
            print(f"   Confirmatory findings: {len(novelty.get('confirmatory_findings', []))}")
            print(f"   Novelty score: {novelty.get('novelty_score', 0):.2f}")
            
            if novelty.get('novel_findings'):
                print(f"\n   🔬 NOVEL DISCOVERIES:")
                for i, finding in enumerate(novelty['novel_findings'][:3], 1):
                    print(f"      {i}. {finding.get('finding', 'Unknown')}: Current={finding.get('current_value', 0):.3f}, Literature={finding.get('literature_mean', 0):.3f}")
        
        # Literature synthesis
        synthesis = results.get('literature_synthesis', {})
        if synthesis:
            print(f"\n📊 LITERATURE SYNTHESIS:")
            
            corr_summary = synthesis.get('correlation_summary', {})
            if corr_summary:
                print(f"   Correlations found: {corr_summary.get('count', 0)}")
                print(f"   Mean correlation: {corr_summary.get('mean', 0):.3f}")
                print(f"   Range: {corr_summary.get('range', [0, 0])}")
            
            sample_summary = synthesis.get('sample_size_summary', {})
            if sample_summary:
                print(f"   Sample sizes: {sample_summary.get('count', 0)} studies")
                print(f"   Median sample: {sample_summary.get('median', 0):,}")
        
        # Validation results
        validation = results.get('validation_results', {})
        if validation:
            print(f"\n✅ VALIDATION RESULTS:")
            
            consistency = validation.get('consistency_analysis', {})
            if consistency:
                is_consistent = consistency.get('consistent', False)
                status = "✅ Consistent" if is_consistent else "⚠️ Inconsistent"
                print(f"   Literature consistency: {status}")
                
            power_comp = validation.get('statistical_power_comparison', {})
            if power_comp:
                percentile = power_comp.get('percentile_rank', 0)
                print(f"   Sample size percentile: {percentile:.1f}% of literature")
            
            recommendations = validation.get('recommendations', [])
            if recommendations:
                print(f"\n   📋 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"      {i}. {rec}")
        
        print("\n" + "="*80)
        print("📚 Literature research complete!")
        print("📁 Results saved to: outputs/literature_research_results.json")
        print("="*80)


def main():
    """Test the literature research agent"""
    agent = LiteratureResearchAgent()
    
    # Mock analysis results for testing
    mock_results = {
        'correlation_analysis': {
            'primary_correlations': {
                'ecog_memory_vs_reaction_time': {
                    'correlation_coefficient': 0.45,
                    'p_value': 0.001,
                    'sample_size': 200
                }
            }
        },
        'data_summary': {
            'baseline_subjects': 150
        }
    }
    
    results = agent.research_findings(mock_results)
    agent.print_research_summary(results)
    return results


if __name__ == "__main__":
    # Add numpy import for testing
    import numpy as np
    main()