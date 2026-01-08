"""
vLLM Client - Simplified client for local vLLM server.
Supports: Qwen 2.5 Coder, DeepSeek R1

Usage:
    from src.llm_clients import VLLMClient
    
    client = VLLMClient()  # Auto-detects running model
    response = client.generate(prompt, system_prompt="You are a helpful assistant")
    print(response.content)
"""
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    finish_reason: str = "stop"
    
    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model_name}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.latency_seconds:.2f}s)"
        )


# =============================================================================
# VLLM CLIENT
# =============================================================================

class VLLMClient:
    """
    Simple client for local vLLM server.
    
    Connects to localhost:8000 by default (standard vLLM port).
    Auto-detects the running model on first use.
    
    Supported models:
    - Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
    - casperhansen/deepseek-r1-distill-qwen-7b-awq
    """
    
    # Default generation parameters
    DEFAULT_TEMPERATURE = 0.0  # Deterministic for reproducibility
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT = 300  # 5 minutes for long generations
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (default: localhost:8000)
            api_key: API key (usually "EMPTY" for local vLLM)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._model_name: Optional[str] = None
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        logger.info(f"VLLMClient initialized -> {self.base_url}")
    
    # =========================================================================
    # MODEL DETECTION
    # =========================================================================
    
    @property
    def model_name(self) -> str:
        """Get the name of the running model (auto-detected)."""
        if self._model_name is None:
            self._model_name = self._detect_model()
        return self._model_name
    
    def _detect_model(self) -> str:
        """
        Detect which model is running on vLLM server.
        
        Returns:
            Model name string
            
        Raises:
            ConnectionError: If vLLM server is not reachable
        """
        try:
            models = self.client.models.list()
            if models.data:
                model_name = models.data[0].id
                logger.info(f"✅ Detected model: {model_name}")
                return model_name
            else:
                raise ConnectionError("No models found on vLLM server")
        except Exception as e:
            logger.error(f"❌ Could not connect to vLLM at {self.base_url}: {e}")
            raise ConnectionError(
                f"vLLM server not reachable at {self.base_url}. "
                "Make sure to run: bash scripts/run_single_model.sh qwen"
            ) from e
    
    def is_deepseek(self) -> bool:
        """Check if running model is DeepSeek (has special thinking tags)."""
        return "deepseek" in self.model_name.lower()
    
    def is_qwen(self) -> bool:
        """Check if running model is Qwen."""
        return "qwen" in self.model_name.lower()
    
    # =========================================================================
    # GENERATION
    # =========================================================================
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to the API
            
        Returns:
            LLMResponse with generated content and metadata
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        latency = time.time() - start_time
        
        # Extract response
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Token usage (some servers may not provide this)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        logger.info(
            f"Generated {completion_tokens} tokens in {latency:.2f}s "
            f"({completion_tokens/latency:.1f} tok/s)" if latency > 0 else ""
        )
        
        return LLMResponse(
            content=content,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            finish_reason=choice.finish_reason or "stop"
        )
    
    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with automatic retry on failure.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
        
        raise RuntimeError(
            f"All {max_retries} attempts failed. Last error: {last_error}"
        )
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate).
        
        Uses ~4 chars per token heuristic. For exact counts,
        would need model-specific tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def check_context_limit(self, text: str, limit: int = 32000) -> bool:
        """
        Check if text fits within context window.
        
        Args:
            text: Input text
            limit: Token limit (default 32k for our models)
            
        Returns:
            True if within limit
        """
        estimated = self.estimate_tokens(text)
        return estimated < limit
    
    def __repr__(self) -> str:
        model = self._model_name or "not_connected"
        return f"VLLMClient(model={model}, url={self.base_url})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_client() -> VLLMClient:
    """
    Get a configured VLLMClient instance.
    
    Returns:
        VLLMClient connected to localhost:8000
    """
    return VLLMClient()


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick generation without creating client explicitly.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        **kwargs: Additional parameters
        
    Returns:
        Generated text content
    """
    client = VLLMClient()
    response = client.generate(prompt, system_prompt, **kwargs)
    return response.content