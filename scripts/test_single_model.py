import sys
import os

# Aggiungiamo la root al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_clients.client_manager import ClientManager

def test_current_model():
    print("="*60)
    print("🧪 TEST: INTERROGATING LOCALHOST:8000")
    print("="*60)

    # Chiediamo un client generico. 
    # Il nome "current_active_model" serve solo per i log, 
    # tanto il manager punta sempre alla porta 8000.
    client = ClientManager.get_client("current_active_model")
    
    # Prompt di test (Coding task)
    prompt = "Write a Python function to check if a number is prime. Optimize for speed."
    
    print(f"📤 Sending prompt: '{prompt}'")
    print("⏳ Waiting for response...")
    
    try:
        response = client.generate(prompt, max_tokens=500, temperature=0.1)
        
        print("\n✅ STATUS: SUCCESS")
        print(f"⏱️ Latency: {response.latency_seconds:.2f}s")
        print(f"📊 Tokens: {response.completion_tokens}")
        print("-" * 30)
        print("📝 MODEL OUTPUT:")
        print(response.content)
        print("-" * 30)
        
        # Check per vedere se è un modello Reasoning (DeepSeek) o Standard (Qwen)
        if "<think>" in response.content:
            print("🧠 DETECTED REASONING MODEL (DeepSeek R1 style)")
        else:
            print("⚡ DETECTED STANDARD MODEL (Qwen/GPT style)")
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print("Suggestion: Check if 'bash scripts/run_single_model.sh' is running.")

if __name__ == "__main__":
    test_current_model()