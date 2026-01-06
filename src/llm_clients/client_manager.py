"""
LLM Client Manager
Centralized management of all LLM clients.
Updated for 7B/8B Green AI Experimentation.
"""
from typing import Dict, Optional
import yaml
from pathlib import Path

from .openai_client import OpenAIClient
from .base_client import BaseLLMClient


class LLMClientManager:
    """
    Manager for all LLM clients.
    """
    
    # Configurazione Modelli per lo Studio Green (Small Models Focus)
    MODEL_CONFIGS = {
        # --- BASELINE (General Purpose) ---
        "llama-3.1-8b": {
            "client_class": OpenAIClient, 
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct", # Nome huggingface
            "provider": "vllm_local"
        },
        
        # --- CODE SPECIALIST (Execution) ---
        "qwen2.5-coder-7b": {
            "client_class": OpenAIClient,
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "provider": "vllm_local"
        },
        "qwen2.5-coder-32b": { # Teniamo anche il 32B come riferimento "Large" se serve
            "client_class": OpenAIClient,
            "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "provider": "vllm_local"
        },

        # --- REASONING SPECIALIST (Debugging/Planning) ---
        "deepseek-r1-7b": {
            "client_class": OpenAIClient,
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "provider": "vllm_local"
        },
        
        # --- PROPRIETARY (Reference Upper Bound) ---
        "gpt-4o-mini": { # Usiamo il mini per confronto costi/green
            "client_class": OpenAIClient,
            "model_name": "gpt-4o-mini",
            "provider": "openai"
        }
    }
    
    def __init__(self, api_keys_path: str = "configs/llm_api_keys.yaml"):
        self.api_keys_path = Path(api_keys_path)
        self.api_keys = self._load_api_keys()
        self.clients: Dict[str, BaseLLMClient] = {}
    
    def _load_api_keys(self) -> Dict:
        if not self.api_keys_path.exists():
            print(f"Warning: {self.api_keys_path} not found. Creating template.")
            self._create_template_config()
            
        with open(self.api_keys_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _create_template_config(self):
        """Creates a template config file"""
        self.api_keys_path.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "openai": {"api_key": "sk-..."},
            "vllm_local": {
                "api_key": "EMPTY", 
                "base_url": "http://localhost:8000/v1" # Porta di default vLLM
            },
            "ollama_local": {
                "api_key": "ollama", 
                "base_url": "http://localhost:11434/v1"
            }
        }
        with open(self.api_keys_path, 'w') as f:
            yaml.dump(template, f)

    def get_client(self, model_short_name: str) -> BaseLLMClient:
        if model_short_name in self.clients:
            return self.clients[model_short_name]
        
        if model_short_name not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Model '{model_short_name}' not configured. Available: {available}")
        
        config = self.MODEL_CONFIGS[model_short_name]
        provider = config["provider"]
        
        # Get credentials
        creds = self.api_keys.get(provider, {})
        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        
        if not api_key:
             # Fallback per vLLM se non c'è chiave esplicita
             if "local" in provider:
                 api_key = "EMPTY"
             else:
                 raise ValueError(f"API key for provider '{provider}' not found.")

        # Initialize
        client_class = config["client_class"]
        
        print(f"🔌 Connecting to {model_short_name} via {provider} at {base_url}...")
        
        client = client_class(
            model_name=config["model_name"],
            api_key=api_key,
            base_url=base_url 
        )
        
        self.clients[model_short_name] = client
        return client