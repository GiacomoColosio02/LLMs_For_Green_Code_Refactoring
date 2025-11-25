"""
Test script for FREE LLM clients only
Tests only models with free tier:
- Gemini 3 Pro (Google)
- Gemma 3 27B (Google)
- Qwen2.5-Coder-32B (Alibaba - free trial)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_clients import LLMClientManager


# Models with FREE tier
FREE_MODELS = {
    "gemini-3-pro": {
        "name": "Gemini 3 Pro",
        "provider": "Google",
        "free_limits": "15 req/min, 1500 req/day"
    },
    "gemma-3-27b": {
        "name": "Gemma 3 27B",
        "provider": "Google",
        "free_limits": "Unlimited (open weights)"
    },
    "qwen2.5-coder-32b": {
        "name": "Qwen2.5-Coder-32B",
        "provider": "Alibaba",
        "free_limits": "Trial credits available"
    }
}


def main():
    """Test only free-tier LLM clients."""
    
    print("\n" + "="*70)
    print("🆓 TESTING FREE LLM CLIENTS ONLY")
    print("="*70)
    print("\n📋 Models to test:")
    for model_id, info in FREE_MODELS.items():
        print(f"  • {info['name']} ({info['provider']})")
        print(f"    Free tier: {info['free_limits']}")
    
    print("\n" + "="*70)
    
    # Initialize manager
    try:
        print("\n🔧 Initializing LLM Client Manager...")
        manager = LLMClientManager(api_keys_path="configs/llm_api_keys.yaml")
        print("✅ Manager initialized")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Create configs/llm_api_keys.yaml with your API keys:")
        print("""
google:
  api_key: "YOUR_GOOGLE_API_KEY"

alibaba:
  api_key: "YOUR_ALIBABA_API_KEY"
        """)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing manager: {e}")
        sys.exit(1)
    
    # Test each free model
    results = {}
    
    for model_id in FREE_MODELS.keys():
        print("\n" + "-"*70)
        print(f"🧪 Testing {FREE_MODELS[model_id]['name']}...")
        print("-"*70)
        
        try:
            # Get client
            print(f"  📡 Connecting to {FREE_MODELS[model_id]['provider']}...")
            client = manager.get_client(model_id)
            
            # Make test request
            print(f"  🔄 Sending test prompt...")
            response = client.generate(
                prompt="Say 'Hello' in one word.",
                temperature=0.0,
                max_tokens=10
            )
            
            # Show results
            print(f"\n  ✅ SUCCESS!")
            print(f"  📝 Response: '{response.content}'")
            print(f"  ⏱️  Latency: {response.latency_seconds:.2f}s")
            print(f"  🔢 Tokens: {response.prompt_tokens} input + {response.completion_tokens} output = {response.total_tokens} total")
            
            results[model_id] = True
            
        except Exception as e:
            print(f"\n  ❌ FAILED: {str(e)}")
            print(f"\n  💡 Troubleshooting:")
            
            if "google" in model_id:
                print(f"     • Get free API key: https://makersuite.google.com/app/apikey")
                print(f"     • Add to configs/llm_api_keys.yaml under google.api_key")
            elif "qwen" in model_id:
                print(f"     • Get free trial: https://dashscope.console.aliyun.com/")
                print(f"     • Add to configs/llm_api_keys.yaml under alibaba.api_key")
            
            results[model_id] = False
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for model_id, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {FREE_MODELS[model_id]['name']}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n  Total: {successful}/{total} clients working")
    
    if successful == total:
        print("\n🎉 All free clients are working!")
        print("\n💡 Next steps:")
        print("   1. You can now test prompt templates")
        print("   2. Or add paid API keys to test all models:")
        print("      python scripts/test_llm_clients.py")
    elif successful > 0:
        print(f"\n⚠️  {successful} client(s) working, {total - successful} failed")
        print("   Fix the failed ones before proceeding")
    else:
        print("\n❌ No clients are working!")
        print("   Check your API keys in configs/llm_api_keys.yaml")
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")
    
    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()