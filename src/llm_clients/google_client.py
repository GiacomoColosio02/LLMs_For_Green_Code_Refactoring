"""
Google Gemini/Gemma Client
"""
from typing import Optional
import time
import google.generativeai as genai

from .base_client import BaseLLMClient, LLMResponse


class GoogleClient(BaseLLMClient):
    """
    Client for Google models.
    
    Supported models:
    - gemini-3-pro
    - gemma-3-27b-it
    """
    
    PROVIDER = "google"
    
    def _initialize_client(self):
        """Initialize Google Generative AI client."""
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Google models.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction (for Gemini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generation config
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
        
        # Call API
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        latency = time.time() - start_time
        
        # Extract response
        content = response.text
        
        # Token counts (approximate for Gemini)
        prompt_tokens = self.count_tokens(full_prompt)
        completion_tokens = self.count_tokens(content)
        
        return LLMResponse(
            model_name=self.model_name,
            provider=self.PROVIDER,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=latency,
            metadata={
                "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
                "safety_ratings": [
                    {
                        "category": rating.category.name,
                        "probability": rating.probability.name
                    }
                    for rating in response.candidates[0].safety_ratings
                ] if response.candidates else []
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Google's method.
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except:
            # Fallback: approximate
            return len(text) // 4