#!/usr/bin/env python3
"""
Token Usage Manager
===================

Monitors and controls AI API token usage across multiple providers.
Prevents exceeding usage quotas and provides cost estimation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TokenUsage:
    """Track token usage for a single API call"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    timestamp: datetime
    request_type: str  # discovery, analysis, literature, etc.

@dataclass
class ProviderLimits:
    """Usage limits for an API provider"""
    daily_token_limit: int
    monthly_token_limit: int
    daily_cost_limit: float
    monthly_cost_limit: float
    requests_per_minute: int
    enabled: bool = True

class TokenManager:
    """
    Comprehensive token usage management system
    
    Features:
    - Multi-provider support (Claude, GPT, Gemini)
    - Real-time usage tracking
    - Configurable limits and alerts
    - Cost estimation
    - Usage analytics
    - Graceful degradation when limits approached
    """
    
    def __init__(self, config_path: str = "config/usage_limits.json"):
        self.config_path = config_path
        self.usage_log_path = "outputs/token_usage.json"
        
        # Token pricing (approximate, update as needed)
        self.pricing = {
            # Anthropic (legacy names)
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},  # per 1K tokens
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            # Anthropic (modern 3.5 family)
            'claude-3-5-sonnet-20240620': {'input': 0.003, 'output': 0.015},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-5-haiku-20241022': {'input': 0.00025, 'output': 0.00125},
            # OpenAI
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            # Google
            'gemini-pro': {'input': 0.00025, 'output': 0.0005}
        }
        
        self.limits = self._load_limits()
        self.usage_history = self._load_usage_history()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_limits(self) -> Dict[str, ProviderLimits]:
        """Load usage limits from configuration"""
        default_limits = {
            'claude': ProviderLimits(
                daily_token_limit=1000000,    # 1M tokens per day
                monthly_token_limit=25000000, # 25M tokens per month
                daily_cost_limit=50.0,        # $50 per day
                monthly_cost_limit=1000.0,    # $1000 per month
                requests_per_minute=60,       # 60 RPM
                enabled=True
            ),
            'openai': ProviderLimits(
                daily_token_limit=500000,     # 500K tokens per day
                monthly_token_limit=10000000, # 10M tokens per month
                daily_cost_limit=100.0,       # $100 per day
                monthly_cost_limit=2000.0,    # $2000 per month
                requests_per_minute=60,       # 60 RPM
                enabled=True
            ),
            'gemini': ProviderLimits(
                daily_token_limit=2000000,    # 2M tokens per day
                monthly_token_limit=50000000, # 50M tokens per month
                daily_cost_limit=25.0,        # $25 per day
                monthly_cost_limit=500.0,     # $500 per month
                requests_per_minute=60,       # 60 RPM
                enabled=True
            )
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                limits = {}
                for provider, limit_data in config.items():
                    limits[provider] = ProviderLimits(**limit_data)
                return limits
            else:
                # Save default configuration
                self._save_limits(default_limits)
                return default_limits
                
        except Exception as e:
            self.logger.warning(f"Error loading limits config: {e}. Using defaults.")
            return default_limits
    
    def _save_limits(self, limits: Dict[str, ProviderLimits]):
        """Save usage limits to configuration file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config = {}
        for provider, limit_obj in limits.items():
            config[provider] = asdict(limit_obj)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_usage_history(self) -> List[TokenUsage]:
        """Load historical token usage"""
        try:
            if os.path.exists(self.usage_log_path):
                with open(self.usage_log_path, 'r') as f:
                    data = json.load(f)
                
                usage_list = []
                for entry in data:
                    entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    usage_list.append(TokenUsage(**entry))
                
                return usage_list
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Error loading usage history: {e}")
            return []

    def _save_usage_history(self):
        """Persist token usage history to outputs/token_usage.json"""
        try:
            os.makedirs(os.path.dirname(self.usage_log_path), exist_ok=True)
            data = []
            for usage in self.usage_history:
                entry = asdict(usage)
                entry['timestamp'] = usage.timestamp.isoformat()
                data.append(entry)
            with open(self.usage_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save token usage history: {e}")
    
    def _match_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Best-effort pricing match: exact key, then family prefix heuristics."""
        if model in self.pricing:
            return self.pricing[model]
        # Heuristics by family
        anthro_map = {
            'claude-3-5-sonnet': 'claude-3-5-sonnet-20240620',
            'claude-3-5-haiku': 'claude-3-5-haiku-20241022',
            'claude-3-sonnet': 'claude-3-sonnet',
            'claude-3-haiku': 'claude-3-haiku'
        }
        for prefix, key in anthro_map.items():
            if model.startswith(prefix):
                return self.pricing.get(key)
        openai_map = {
            'gpt-4o': 'gpt-4o',
            'gpt-4': 'gpt-4',
            'gpt-3.5': 'gpt-3.5-turbo'
        }
        for prefix, key in openai_map.items():
            if model.startswith(prefix):
                return self.pricing.get(key)
        if model.startswith('gemini-pro'):
            return self.pricing.get('gemini-pro')
        return None
    
    def estimate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage"""
        pricing = self._match_pricing(model)
        if pricing:
            input_cost = (input_tokens / 1000) * pricing['input']
            output_cost = (output_tokens / 1000) * pricing['output']
            return input_cost + output_cost
        else:
            # Default estimation
            return (input_tokens + output_tokens) / 1000 * 0.002
    
    def check_limits(self, provider: str, estimated_tokens: int) -> Tuple[bool, str]:
        """
        Check if request would exceed limits
        
        Returns:
        (can_proceed, reason_if_not)
        """
        if provider not in self.limits or not self.limits[provider].enabled:
            return False, f"Provider {provider} not configured or disabled"
        
        limits = self.limits[provider]
        now = datetime.now()
        
        # Calculate current usage
        daily_usage = self._get_usage_in_period(provider, now - timedelta(days=1), now)
        monthly_usage = self._get_usage_in_period(provider, now - timedelta(days=30), now)
        
        # Check token limits
        if daily_usage['tokens'] + estimated_tokens > limits.daily_token_limit:
            return False, f"Would exceed daily token limit ({daily_usage['tokens']:,} + {estimated_tokens:,} > {limits.daily_token_limit:,})"
        
        if monthly_usage['tokens'] + estimated_tokens > limits.monthly_token_limit:
            return False, f"Would exceed monthly token limit ({monthly_usage['tokens']:,} + {estimated_tokens:,} > {limits.monthly_token_limit:,})"
        
        # Check cost limits (rough estimate)
        estimated_cost = estimated_tokens / 1000 * 0.005  # Conservative estimate
        
        if daily_usage['cost'] + estimated_cost > limits.daily_cost_limit:
            return False, f"Would exceed daily cost limit (${daily_usage['cost']:.2f} + ${estimated_cost:.2f} > ${limits.daily_cost_limit:.2f})"
        
        if monthly_usage['cost'] + estimated_cost > limits.monthly_cost_limit:
            return False, f"Would exceed monthly cost limit (${monthly_usage['cost']:.2f} + ${estimated_cost:.2f} > ${limits.monthly_cost_limit:.2f})"
        
        return True, "OK"
    
    def _get_usage_in_period(self, provider: str, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Get total usage for provider in time period"""
        usage = {'tokens': 0, 'cost': 0.0, 'requests': 0}
        
        for entry in self.usage_history:
            if (entry.provider == provider and 
                start_time <= entry.timestamp <= end_time):
                usage['tokens'] += entry.total_tokens
                usage['cost'] += entry.cost_estimate
                usage['requests'] += 1
        
        return usage
    
    def log_usage(self, provider: str, model: str, input_tokens: int, 
                  output_tokens: int, request_type: str = "general") -> TokenUsage:
        """Log token usage for a request"""
        total_tokens = input_tokens + output_tokens
        cost_estimate = self.estimate_cost(provider, model, input_tokens, output_tokens)
        
        usage = TokenUsage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            timestamp=datetime.now(),
            request_type=request_type
        )
        
        self.usage_history.append(usage)
        self._save_usage_history()
        
        # Log to console
        self.logger.info(f"Token usage logged: {provider}/{model} - {total_tokens:,} tokens (${cost_estimate:.4f}) - {request_type}")
        
        return usage
    
    def get_usage_summary(self, days: int = 30) -> Dict[str, any]:
        """Get usage summary for the past N days"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        summary = {
            'period_days': days,
            'start_date': start_time.isoformat(),
            'end_date': end_time.isoformat(),
            'by_provider': {},
            'by_request_type': {},
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_requests': 0
        }
        
        # Calculate by provider
        for provider in ['claude', 'openai', 'gemini']:
            usage = self._get_usage_in_period(provider, start_time, end_time)
            if usage['requests'] > 0:
                summary['by_provider'][provider] = usage
                summary['total_tokens'] += usage['tokens']
                summary['total_cost'] += usage['cost']
                summary['total_requests'] += usage['requests']
        
        # Calculate by request type
        request_types = {}
        for entry in self.usage_history:
            if start_time <= entry.timestamp <= end_time:
                if entry.request_type not in request_types:
                    request_types[entry.request_type] = {'tokens': 0, 'cost': 0.0, 'requests': 0}
                
                request_types[entry.request_type]['tokens'] += entry.total_tokens
                request_types[entry.request_type]['cost'] += entry.cost_estimate
                request_types[entry.request_type]['requests'] += 1
        
        summary['by_request_type'] = request_types
        
        return summary
    
    def get_current_limits_status(self) -> Dict[str, Dict[str, float]]:
        """Get current usage vs limits for all providers"""
        now = datetime.now()
        status = {}
        
        for provider, limits in self.limits.items():
            if not limits.enabled:
                continue
                
            daily_usage = self._get_usage_in_period(provider, now - timedelta(days=1), now)
            monthly_usage = self._get_usage_in_period(provider, now - timedelta(days=30), now)
            
            status[provider] = {
                'daily_tokens_used': daily_usage['tokens'],
                'daily_tokens_limit': limits.daily_token_limit,
                'daily_tokens_percent': (daily_usage['tokens'] / limits.daily_token_limit * 100) if limits.daily_token_limit > 0 else 0,
                
                'monthly_tokens_used': monthly_usage['tokens'],
                'monthly_tokens_limit': limits.monthly_token_limit,
                'monthly_tokens_percent': (monthly_usage['tokens'] / limits.monthly_token_limit * 100) if limits.monthly_token_limit > 0 else 0,
                
                'daily_cost_used': daily_usage['cost'],
                'daily_cost_limit': limits.daily_cost_limit,
                'daily_cost_percent': (daily_usage['cost'] / limits.daily_cost_limit * 100) if limits.daily_cost_limit > 0 else 0,
                
                'monthly_cost_used': monthly_usage['cost'],
                'monthly_cost_limit': limits.monthly_cost_limit,
                'monthly_cost_percent': (monthly_usage['cost'] / limits.monthly_cost_limit * 100) if limits.monthly_cost_limit > 0 else 0,
            }
        
        return status
    
    def suggest_provider(self, estimated_tokens: int) -> Optional[str]:
        """Suggest best provider based on current usage and limits"""
        candidates = []
        
        for provider in ['claude', 'openai', 'gemini']:
            can_proceed, reason = self.check_limits(provider, estimated_tokens)
            if can_proceed:
                # Calculate "pressure" on limits
                status = self.get_current_limits_status().get(provider, {})
                token_pressure = status.get('daily_tokens_percent', 0) + status.get('monthly_tokens_percent', 0)
                cost_pressure = status.get('daily_cost_percent', 0) + status.get('monthly_cost_percent', 0)
                total_pressure = token_pressure + cost_pressure
                
                candidates.append((provider, total_pressure))
        
        if not candidates:
            return None
        
        # Return provider with lowest pressure
        return min(candidates, key=lambda x: x[1])[0]
    
    def print_usage_report(self):
        """Print a formatted usage report"""
        print("\n" + "="*60)
        print("🔍 TOKEN USAGE REPORT")
        print("="*60)
        
        # Current limits status
        status = self.get_current_limits_status()
        
        for provider, provider_status in status.items():
            print(f"\n📊 {provider.upper()} Usage:")
            print(f"   Daily Tokens:  {provider_status['daily_tokens_used']:,} / {provider_status['daily_tokens_limit']:,} ({provider_status['daily_tokens_percent']:.1f}%)")
            print(f"   Monthly Tokens: {provider_status['monthly_tokens_used']:,} / {provider_status['monthly_tokens_limit']:,} ({provider_status['monthly_tokens_percent']:.1f}%)")
            print(f"   Daily Cost:    ${provider_status['daily_cost_used']:.2f} / ${provider_status['daily_cost_limit']:.2f} ({provider_status['daily_cost_percent']:.1f}%)")
            print(f"   Monthly Cost:  ${provider_status['monthly_cost_used']:.2f} / ${provider_status['monthly_cost_limit']:.2f} ({provider_status['monthly_cost_percent']:.1f}%)")
        
        # Recent activity
        summary = self.get_usage_summary(7)  # Last 7 days
        print(f"\n📈 Last 7 Days Summary:")
        print(f"   Total Requests: {summary['total_requests']:,}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Total Cost: ${summary['total_cost']:.2f}")
        
        if summary['by_request_type']:
            print(f"\n🎯 By Request Type:")
            for req_type, data in summary['by_request_type'].items():
                print(f"   {req_type}: {data['requests']} requests, {data['tokens']:,} tokens, ${data['cost']:.2f}")