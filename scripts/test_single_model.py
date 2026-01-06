import sys
import os

# Aggiunge la root al path per importare i moduli src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_clients.client_manager import ClientManager

def test_current_model():
    print("="*60)
    print("🧪 TEST: INTERROGATING LOCALHOST:8000")
    print("="*60)

    # Il manager punta sempre alla porta 8000
    client = ClientManager.get_client("active_model")

    # Prompt semplice di coding
    prompt = "Write a Python function to check if a number is prime. Optimize for speed."

    print(f"📤 Sending prompt: '{prompt}'")
    print("⏳ Waiting for response...")

    try:
        # Temperature 0.1 per codice preciso
        response = client.generate(prompt, max_tokens=500, temperature=0.1)

        print("\n✅ STATUS: SUCCESS")
        print(f"⏱️ Latency: {response.latency_seconds:.2f}s")
        print("-" * 30)
        print("📝 MODEL OUTPUT:")
        print(response.content[:500] + "...") # Stampa i primi 500 caratteri
        print("-" * 30)

        if "<think>" in response.content:
            print("🧠 DETECTED REASONING MODEL (DeepSeek R1)")
        else:
            print("⚡ DETECTED STANDARD MODEL (Qwen Coder)")

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

if __name__ == "__main__":
    test_current_model()