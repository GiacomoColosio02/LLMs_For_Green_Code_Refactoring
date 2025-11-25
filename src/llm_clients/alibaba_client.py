"""
Alibaba Qwen Client
"""
from typing import Optional
import time
import dashscope
from dashscope import Generation

from .base_client import BaseLLMClient, LLMResponse


class AlibabaClient(BaseLLMClient):
    """
    Client for Alibaba Qwen models.
    
    Supported models:
    - qwen2.5-coder-32b-instruct
    - qwen3-32b
    - qwq-32b-preview
    """
    
    PROVIDER = "alibaba"
    
    def _initialize_client(self):
        """Initialize Dashscope (Alibaba Cloud) client."""
        dashscope.api_key = self.api_key
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Qwen models.
        
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
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        # Call API
        response = Generation.call(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message',  # Get structured output
            **kwargs
        )
        
        latency = time.time() - start_time
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Qwen API error: {response.code} - {response.message}")
        
        # Extract response
        content = response.output.choices[0].message.content
        
        # Token usage
        usage = response.usage
        
        return LLMResponse(
            model_name=self.model_name,
            provider=self.PROVIDER,
            content=content,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_seconds=latency,
            metadata={
                "finish_reason": response.output.choices[0].finish_reason,
                "request_id": response.request_id
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens for Qwen.
        Qwen uses similar tokenizer to GPT, approximate.
        """
        return len(text) // 4