import time
import os
import re
import subprocess
from openai import OpenAI
from multiprocessing import Process, Queue

# --- Configuration ---
MODEL = "llama3.2:1b"
OLLAMA_API_URL = "http://localhost:11434/v1" 

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CURRENT_DIR, "build")
INGEST_BIN = os.path.join(BUILD_DIR, "agent_ingest")
QUERY_BIN = os.path.join(BUILD_DIR, "agent_query")

# 1. Local Tool
def run_cpp_binary_tool(binary_path: str, agent_name: str) -> str:
    print(f"    [Core] {agent_name} Tool: Launching binary -> {binary_path}")
    if not os.path.exists(binary_path):
        return f"Error: Binary not found at {binary_path}"
    try:
        # stdout=None 允许 C++ 的输出直接打印到屏幕上
        subprocess.run([binary_path], check=True, text=True)
        return "Binary Execution Success"
    except subprocess.CalledProcessError as e:
        return f"Binary Failed (Exit Code {e.returncode})"
    except Exception as e:
        return f"Exception: {str(e)}"

# 2. Ingest Agent
def ingest_agent_process(name, binary_path, output_queue):
    print(f" [Agent {name}] Started (PID: {os.getpid()})")
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    prompt = f"""Task: Check if binary path is valid.
Binary Path: "{binary_path}"

Rules:
1. Unless empty, reply YES.
2. We MUST execute this.
Answer:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        decision = re.sub(r'[^A-Z]', '', response.choices[0].message.content.strip().upper())
        print(f"    [Agent {name}] LLM Decision: {decision}")
        
        if "" != decision:
            status = run_cpp_binary_tool(binary_path, name)
            output_queue.put(f"[ACCEPTED] {name}: {status}")
        else:
            output_queue.put(f"[DECLINED] {name}: Logic decided not to run.")
            
    except Exception as e:
        output_queue.put(f"[ERROR] {name}: {e}")

# 3. Query Agent
def query_agent_process(name, binary_path, output_queue):
    print(f" [Agent {name}] Started (PID: {os.getpid()})")
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    prompt = f"""Task: Check if binary path is valid.
Binary Path: "{binary_path}"

Rules:
1. Unless empty, reply YES.
2. We MUST execute this.
Answer:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        decision = re.sub(r'[^A-Z]', '', response.choices[0].message.content.strip().upper())
        print(f"    [Agent {name}] LLM Decision: {decision}")
        
        if "" != decision:
            status = run_cpp_binary_tool(binary_path, name)
            output_queue.put(f"[ACCEPTED] {name}: {status}")
        else:
            output_queue.put(f"[DECLINED] {name}: Logic decided not to run.")
            
    except Exception as e:
        output_queue.put(f"[ERROR] {name}: {e}")

# 4. Aggregator Agent (Audit Mode)
def aggregator_agent(ingest_status, query_status):
    print(f"\n [Aggregator Agent] performing audit...")
    print(f"   Ingest Agent: {ingest_status}")
    print(f"   Query Agent:  {query_status}")    
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    # [关键修改] 要求 LLM 明确陈述“做了什么” (Action Taken)
    prompt = f"""You are a Pipeline Auditor.
    
Agent Reports:
Ingest Agent: {ingest_status}
Query Agent: {query_status}

Give a one sentence summary of the action taken by the pipeline based on the reports above.

Output:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# --- Main Program ---
if __name__ == "__main__":
    if not os.path.exists(INGEST_BIN) or not os.path.exists(QUERY_BIN):
        print("Error: Binaries not found.")
        exit(1)

    print(f" Task: Parallel AdaDisk Workflow")
    q_ingest = Queue()
    q_query = Queue()
    
    p1 = Process(target=ingest_agent_process, args=("IngestAgent", INGEST_BIN, q_ingest))
    p2 = Process(target=query_agent_process, args=("QueryAgent", QUERY_BIN, q_query))
    
    p1.start()
    p2.start()
    
    res_ingest = q_ingest.get()
    res_query = q_query.get()
    
    p1.join()
    p2.join()
    
    print("\n--- Parallel Execution Complete ---")
    
    # 输出详细审计报告
    final_audit = aggregator_agent(res_ingest, res_query)
    print(f"\n{final_audit}")