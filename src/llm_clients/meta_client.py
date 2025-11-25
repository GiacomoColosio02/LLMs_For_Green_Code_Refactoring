"""
Meta Llama 4 Client (via Together.ai)
"""
from typing import Optional
import time
from together import Together

from .base_client import BaseLLMClient, LLMResponse


class MetaClient(BaseLLMClient):
    """
    Client for Meta Llama models via Together.ai.
    
    Supported models:
    - meta-llama/Llama-4-Maverick-17B-128E-Instruct
    - meta-llama/Llama-4-Scout-17B-16E-Instruct
    
    Note: Requires Together.ai API key
    """
    
    PROVIDER = "meta"
    
    def _initialize_client(self):
        """Initialize Together.ai client for Llama 4."""
        self.client = Together(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Llama 4.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        latency = time.time() - start_time
        
        # Extract response
        content = response.choices[0].message.content
        
        return LLMResponse(
            model_name=self.model_name,
            provider=self.PROVIDER,
            content=content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_seconds=latency,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "model_used": response.model
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens for Llama.
        Llama uses similar tokenizer to GPT, approximate.
        """
        return len(text) // 4