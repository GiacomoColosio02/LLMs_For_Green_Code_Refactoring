"""
OpenAI-compatible client implementation for local vLLM.
"""
import time
import logging
from typing import Optional, Dict, Any
from openai import OpenAI, APIConnectionError, APITimeoutError

from .base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI-compatible APIs (specifically local vLLM).
    Handles port routing via base_url passed in kwargs.
    """
    
    def _initialize_client(self):
        """
        Initialize the OpenAI client with the specific base_url for the model.
        """
        # Estraiamo base_url dai kwargs, default a porta 8000
        self.base_url = self.kwargs.get("base_url", "http://localhost:8000/v1")
        
        logger.info(f"Initializing OpenAIClient for {self.model_name} at {self.base_url}")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key if self.api_key else "EMPTY"
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
        Generate text using the local vLLM server.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        
        try:
            # Chiamata all'API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs # Passa eventuali altri parametri supportati
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Estrazione dati
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Gestione sicura dei token usage (alcuni server proxy potrebbero non inviarli)
            usage = response.usage
            p_tokens = usage.prompt_tokens if usage else 0
            c_tokens = usage.completion_tokens if usage else 0
            t_tokens = usage.total_tokens if usage else 0

            return LLMResponse(
                model_name=self.model_name,
                provider="vllm_local",
                content=content,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                latency_seconds=latency,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "base_url": self.base_url
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with {self.model_name}: {e}")
            raise e

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count. 
        For local logic without loading heavy tokenizers, we use a heuristic 
        or we could use tiktoken if installed.
        """
        # Stima approssimativa (4 caratteri ~= 1 token) per evitare overhead
        # Se necessario, possiamo integrare 'tiktoken' o 'transformers' qui.
        return len(text) // 4