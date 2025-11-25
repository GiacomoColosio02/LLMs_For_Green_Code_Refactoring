"""
Anthropic Claude Opus 4.5 Client
"""
from typing import Optional
import time
from anthropic import Anthropic

from .base_client import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic Claude models.
    
    Supported models:
    - claude-opus-4-5-20251101
    - claude-opus-4-1
    - claude-sonnet-4-5
    """
    
    PROVIDER = "anthropic"
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        self.client = Anthropic(
            api_key=self.api_key,
            timeout=self.timeout
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Claude.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters
        
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Build request
        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Call API
        response = self.client.messages.create(**request_params)
        
        latency = time.time() - start_time
        
        # Extract response
        content = response.content[0].text
        
        return LLMResponse(
            model_name=self.model_name,
            provider=self.PROVIDER,
            content=content,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_seconds=latency,
            metadata={
                "stop_reason": response.stop_reason,
                "model_used": response.model,
                "stop_sequence": response.stop_sequence
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens (Anthropic uses similar tokenizer to GPT).
        Approximation: ~4 chars per token.
        """
        return len(text) // 4