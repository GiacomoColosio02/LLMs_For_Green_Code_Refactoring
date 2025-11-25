"""
LLM Client Manager
Centralized management of all LLM clients
"""
from typing import Dict, Optional
import yaml
from pathlib import Path

from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient
from .alibaba_client import AlibabaClient
from .meta_client import MetaClient
from .base_client import BaseLLMClient


class LLMClientManager:
    """
    Manager for all LLM clients.
    Handles initialization, configuration, and access to all models.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-5": {
            "client_class": OpenAIClient,
            "model_name": "gpt-5",
            "provider": "openai"
        },
        "claude-opus-4.5": {
            "client_class": AnthropicClient,
            "model_name": "claude-opus-4-5-20251101",
            "provider": "anthropic"
        },
        "gemini-3-pro": {
            "client_class": GoogleClient,
            "model_name": "gemini-3-pro",
            "provider": "google"
        },
        "qwen2.5-coder-32b": {
            "client_class": AlibabaClient,
            "model_name": "qwen2.5-coder-32b-instruct",
            "provider": "alibaba"
        },
        "llama-4-maverick": {
            "client_class": MetaClient,
            "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "provider": "meta"
        },
        "gemma-3-27b": {
            "client_class": GoogleClient,
            "model_name": "gemma-3-27b-it",
            "provider": "google"
        }
    }
    
    def __init__(self, api_keys_path: str = "configs/llm_api_keys.yaml"):
        """
        Initialize client manager.
        
        Args:
            api_keys_path: Path to YAML file with API keys
        """
        self.api_keys_path = Path(api_keys_path)
        self.api_keys = self._load_api_keys()
        self.clients: Dict[str, BaseLLMClient] = {}
    
    def _load_api_keys(self) -> Dict:
        """Load API keys from YAML file."""
        if not self.api_keys_path.exists():
            raise FileNotFoundError(
                f"API keys file not found: {self.api_keys_path}\n"
                "Please create configs/llm_api_keys.yaml with your API keys."
            )
        
        with open(self.api_keys_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_client(self, model_short_name: str) -> BaseLLMClient:
        """
        Get or create client for a model.
        
        Args:
            model_short_name: Short name like 'gpt-5', 'claude-opus-4.5', etc.
        
        Returns:
            Initialized LLM client
        
        Raises:
            ValueError: If model not found or API key missing
        """
        # Return cached client if exists
        if model_short_name in self.clients:
            return self.clients[model_short_name]
        
        # Check if model exists
        if model_short_name not in self.MODEL_CONFIGS:
            available = ", ".join(self.MODEL_CONFIGS.keys())
            raise ValueError(
                f"Model '{model_short_name}' not found.\n"
                f"Available models: {available}"
            )
        
        config = self.MODEL_CONFIGS[model_short_name]
        provider = config["provider"]
        
        # Get API key
        provider_keys = self.api_keys.get(provider, {})
        
        # Handle different key names
        if provider == "openai":
            api_key = provider_keys.get("api_key")
        elif provider == "anthropic":
            api_key = provider_keys.get("api_key")
        elif provider == "google":
            api_key = provider_keys.get("api_key")
        elif provider == "alibaba":
            api_key = provider_keys.get("api_key")
        elif provider == "meta":
            # Try different platforms
            api_key = (
                provider_keys.get("together_api_key") or
                provider_keys.get("replicate_api_key") or
                provider_keys.get("huggingface_token")
            )
        else:
            api_key = None
        
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{provider}'.\n"
                f"Please add it to {self.api_keys_path}"
            )
        
        # Initialize client
        client_class = config["client_class"]
        model_name = config["model_name"]
        
        client = client_class(
            model_name=model_name,
            api_key=api_key,
            timeout=self.api_keys.get("timeout_seconds", 120),
            max_retries=self.api_keys.get("max_retries", 3)
        )
        
        # Cache client
        self.clients[model_short_name] = client
        
        print(f"✅ Initialized {model_short_name} ({provider})")
        
        return client
    
    def get_all_models(self) -> list:
        """Get list of all available model names."""
        return list(self.MODEL_CONFIGS.keys())
    
    def test_client(self, model_short_name: str) -> bool:
        """
        Test if a client works.
        
        Args:
            model_short_name: Model to test
        
        Returns:
            True if test successful, False otherwise
        """
        try:
            client = self.get_client(model_short_name)
            response = client.generate(
                prompt="Say 'Hello' in one word.",
                temperature=0.0,
                max_tokens=10
            )
            print(f"✅ {model_short_name} test successful: '{response.content}'")
            return True
        except Exception as e:
            print(f"❌ {model_short_name} test failed: {str(e)}")
            return False
    
    def test_all_clients(self) -> Dict[str, bool]:
        """
        Test all clients.
        
        Returns:
            Dict mapping model name to success status
        """
        results = {}
        print("\n" + "="*60)
        print("🧪 TESTING ALL LLM CLIENTS")
        print("="*60)
        
        for model_name in self.MODEL_CONFIGS.keys():
            print(f"\nTesting {model_name}...")
            results[model_name] = self.test_client(model_name)
        
        print("\n" + "="*60)
        print("📊 RESULTS:")
        successful = sum(results.values())
        total = len(results)
        print(f"✅ {successful}/{total} clients working")
        print("="*60)
        
        return results