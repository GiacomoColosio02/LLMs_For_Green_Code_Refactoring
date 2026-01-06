"""
Factory for LLM Clients.
Routes requests to the correct local vLLM port based on the model name.
"""
import logging
import os
from typing import Dict, Optional
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class ClientManager:
    """
    Manages connections to local vLLM instances.
    """
    
    # Configurazione Porte: Qwen (Coder) -> 8000, DeepSeek (Reasoner) -> 8001
    MODEL_CONFIG = {
        # PORT 8000: The Coder (Qwen)
        "qwen": {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "port": 8000
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "port": 8000
        },
        
        # PORT 8001: The Reasoner (DeepSeek)
        "deepseek": {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "port": 8001
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "port": 8001
        }
    }

    @staticmethod
    def get_client(model_identifier: str, **kwargs) -> BaseLLMClient:
        """
        Returns the correct client configured for the specific local port.
        
        Args:
            model_identifier: 'qwen', 'deepseek', or full HuggingFace ID
            **kwargs: Additional args passed to client constructor
        """
        # Normalizza la chiave
        config = ClientManager.MODEL_CONFIG.get(model_identifier)
        
        # Default API Key per vLLM è spesso "EMPTY" o ignorata
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        
        if not config:
            logger.warning(f"Model '{model_identifier}' not mapped. Defaulting to localhost:8000")
            return OpenAIClient(
                model_name=model_identifier, 
                api_key=api_key,
                base_url="http://localhost:8000/v1",
                **kwargs
            )
            
        port = config["port"]
        full_name = config["name"]
        base_url = f"http://localhost:{port}/v1"
        
        logger.info(f"Routing model '{model_identifier}' -> {base_url}")
        
        return OpenAIClient(
            model_name=full_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )