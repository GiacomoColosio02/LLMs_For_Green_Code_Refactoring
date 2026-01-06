import sys
import os
from openai import OpenAI # Usiamo questo per chiedere la lista modelli

# Aggiunge la root al path per importare i moduli src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_clients.client_manager import ClientManager

def test_current_model():
    print("="*60)
    print("🧪 TEST: INTERROGATING LOCALHOST:8000")
    print("="*60)

    try:
        # 1. AUTO-DETECT: Chiediamo a vLLM quale modello sta girando
        # Questo passaggio è fondamentale perché vLLM rifiuta richieste con nomi sbagliati
        temp_client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        models_list = temp_client.models.list()
        
        if not models_list.data:
            print("❌ ERROR: No models found running on localhost:8000")
            return
            
        real_model_name = models_list.data[0].id
        print(f"🕵️ AUTO-DETECTED MODEL: {real_model_name}")
        
        # 2. ORA chiediamo il client usando il nome VERO
        client = ClientManager.get_client(real_model_name)
        
        # Prompt di test
        prompt = "Write a Python function to check if a number is prime. Optimize for speed."
        
        print(f"📤 Sending prompt: '{prompt}'")
        print("⏳ Waiting for response...")
        
        # Temperature 0.1 per codice preciso
        response = client.generate(prompt, max_tokens=500, temperature=0.1)
        
        print("\n✅ STATUS: SUCCESS")
        print(f"⏱️ Latency: {response.latency_seconds:.2f}s")
        print("-" * 30)
        print("📝 MODEL OUTPUT:")
        # Stampa i primi 500 caratteri puliti
        print(response.content[:500] + "...") 
        print("-" * 30)
        
        if "<think>" in response.content:
            print("🧠 DETECTED REASONING MODEL (DeepSeek R1)")
        else:
            print("⚡ DETECTED STANDARD MODEL (Qwen Coder)")
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print("\nSuggestion: Check if 'bash scripts/run_single_model.sh' is running.")

if __name__ == "__main__":
    test_current_model()