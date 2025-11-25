"""
OpenAI GPT-5 Client
"""
from typing import Optional
import time
from openai import OpenAI
import tiktoken

from .base_client import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI GPT-5 models.
    
    Supported models:
    - gpt-5
    - gpt-5.1
    - gpt-5-mini
    - gpt-5-nano
    """
    
    PROVIDER = "openai"
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            organization=self.kwargs.get('organization', None)
        )
        
        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4/5)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using OpenAI GPT-5.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions for model)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters (top_p, frequency_penalty, etc.)
        
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
                "model_used": response.model,
                "system_fingerprint": getattr(response, 'system_fingerprint', None)
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI tokenizer."""
        return len(self.tokenizer.encode(text))