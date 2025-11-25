"""
Test script for all LLM clients
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_clients import LLMClientManager


def main():
    """Test all LLM clients."""
    
    # Initialize manager
    print("🔧 Initializing LLM Client Manager...")
    manager = LLMClientManager(api_keys_path="configs/llm_api_keys.yaml")
    
    # List available models
    print("\n📋 Available models:")
    for model in manager.get_all_models():
        print(f"  - {model}")
    
    # Test all clients
    results = manager.test_all_clients()
    
    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()