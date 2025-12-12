import time
import os
import re
from openai import OpenAI  # <--- Core change: Using generic standard library
from multiprocessing import Process, Queue

# --- Configuration ---
MODEL = "llama3.2:1b"
# Pointing to local Ollama API address (standard port)
OLLAMA_API_URL = "http://localhost:11434/v1" 

# 1. Local Tool (Unchanged)
def local_sum_tool(data: list) -> int:
    pid = os.getpid()
    print(f"    [Core/PID {pid}] Local Tool: Calculating {data} (sleep 2s)...")
    time.sleep(2) 
    return sum(data)

# 2. Worker Agent (Rewritten for OpenAI SDK calls)
def worker_agent(name, data_chunk, output_queue):
    print(f" [Agent {name}] Started (PID: {os.getpid()})")
    
    # --- Key Change 1: Initialize OpenAI client within the process ---
    # base_url points to Ollama, api_key can be anything (required field)
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    prompt = f"""Task: Check if the input contains numbers.
Input: {data_chunk}

Rules:
1. If it contains numbers, reply ONLY with the word: YES
2. Do not write any other words. Do not explain.

Answer:"""

    try:
        # --- Key Change 2: Use standard chat.completions.create ---
        # Previous: ollama.chat(..., options={'temperature': 0})
        # Now: Written as below, only base_url needs changing for DeepSeek/GPT-4
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0  # Parameters passed directly as keyword arguments
        )
        
        # --- Key Change 3: Parse object response ---
        # Previous: response['message']['content'] (Dictionary style)
        # Now: response.choices[0].message.content (Object style)
        decision = response.choices[0].message.content.strip().upper()
        
        # Clean result
        clean_decision = re.sub(r'[^A-Z]', '', decision)
        print(f"    [Agent {name}] Decision Output: {clean_decision}")
        
        if "YES" in clean_decision:
            result = local_sum_tool(data_chunk)
            output_queue.put(result)
        else:
            print(f"    [Agent {name}] Refused calculation")
            output_queue.put(0)
            
    except Exception as e:
        print(f"    [Agent {name}] Crashed: {e}")
        output_queue.put(0)

# 3. Aggregator Agent (Rewritten for OpenAI SDK calls)
def aggregator_agent(result_a, result_b):
    print(f"\n [Agent Aggregator] Started...")
    
    true_total = result_a + result_b
    
    # Initialize client here as well
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    prompt = f"""You are a helpful assistant.
The calculation is finished.
Worker A result: {result_a}
Worker B result: {result_b}
Total Sum: {true_total}

Please write a very short sentence to report this Total Sum.
Sentence:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7 
    )
    
    return response.choices[0].message.content.strip()

# --- Main Program ---
if __name__ == "__main__":
    data_a = [10, 20, 30]
    data_b = [40, 50, 60]
    
    print(f" Task: A={data_a}, B={data_b}")
    
    q_a = Queue()
    q_b = Queue()
    
    start_time = time.time()
    
    print("\n--- Starting Parallel Calculation (OpenAI SDK Compatible) ---")
    p1 = Process(target=worker_agent, args=("A", data_a, q_a))
    p2 = Process(target=worker_agent, args=("B", data_b, q_b))
    
    p1.start()
    p2.start()
    
    res_a = q_a.get()
    res_b = q_b.get()
    
    p1.join()
    p2.join()
    
    duration = time.time() - start_time
    
    print(f"--- Parallel finished (Duration: {duration:.2f}s) ---")
    
    final_report = aggregator_agent(res_a, res_b)
    print(f"\n Final Report: {final_report}")
    print(f" Python Verification: {sum(data_a + data_b)}")