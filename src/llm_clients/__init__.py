"""
LLM Clients module.

Simplified for local vLLM usage with Qwen and DeepSeek models.

Usage:
    from src.llm_clients import VLLMClient, get_client
    
    # Option 1: Create client explicitly
    client = VLLMClient()
    response = client.generate("Optimize this code...", system_prompt="You are an expert...")
    print(response.content)
    
    # Option 2: Quick generation
    from src.llm_clients import generate
    text = generate("Hello, who are you?")
"""
from .vllm_client import VLLMClient, LLMResponse, get_client, generate

__all__ = [
    "VLLMClient",
    "LLMResponse", 
    "get_client",
    "generate"
]