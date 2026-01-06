"""
Factory for LLM Clients.
SIMPLIFIED VERSION: Connects to whatever model is running on localhost:8000.
Designed for the Single-Model architecture (High Precision FP16).
"""
import logging
import os
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class ClientManager:
    """
    Manages connections to local vLLM.
    
    WARNING: This manager assumes you have launched the correct model 
    using 'scripts/run_single_model.sh'. It always points to port 8000.
    """
    
    @staticmethod
    def get_client(model_identifier: str, **kwargs) -> BaseLLMClient:
        """
        Returns a client pointing to localhost:8000.
        
        Args:
            model_identifier: Used for logging and metadata (e.g., 'qwen', 'deepseek').
            **kwargs: Additional args passed to client constructor.
        """
        # Default API Key per vLLM (spesso ignorata in locale, ma richiesta dalla lib)
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        
        # In modalità sequenziale, usiamo sempre la porta standard 8000
        base_url = "http://localhost:8000/v1"
        
        logger.info(f"Connecting client for '{model_identifier}' -> {base_url}")
        
        return OpenAIClient(
            model_name=model_identifier, # Questo nome appare nei log della risposta
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )