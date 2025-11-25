"""
Base abstract class for all LLM clients.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    model_name: str
    provider: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    metadata: Dict[str, Any]


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM clients must implement this interface for consistency.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs
        
        # Initialize provider-specific client
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            LLMResponse object with generated text and metadata
        """
        pass
    
    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with automatic retry on failure.
        
        Args:
            Same as generate()
            
        Returns:
            LLMResponse object
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        raise Exception(f"All {self.max_retries} attempts failed. Last error: {last_exception}")
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (provider-specific tokenizer).
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
        