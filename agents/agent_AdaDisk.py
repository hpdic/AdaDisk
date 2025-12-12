#!/usr/bin/env python3
# ==============================================================================
# File: agents/agent_AdaDisk.py
# Project: AdaDisk - Distributed Agentic System for Adaptive RAG
#
# Description:
#   This script serves as the main orchestrator for the AdaDisk system.
#   It coordinates parallel Ingest and Query agents to decouple high-throughput
#   data ingestion from low-latency serving. The system utilizes LLM-driven
#   control planes to make adaptive scheduling decisions and performs
#   system-wide auditing via an Aggregator Agent.
#
#   Usage:
#       python3 agent_AdaDisk.py --model llama3.2:1b
#
# Author: Dongfang Zhao <dzhao@uw.edu>
# Date:   December 12, 2025
#
# Copyright (c) 2025 Dongfang Zhao. All rights reserved.
# ==============================================================================

import time
import os
import re
import subprocess
import argparse
from openai import OpenAI
from multiprocessing import Process, Queue

# --- Configuration ---
DEFAULT_MODEL = "llama3.2:1b" # Default model if no argument is provided
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
        # Use subprocess to run the C++ binary.
        # check=True raises an exception if the exit code is non-zero.
        # text=True allows stdout to pass through to the console as string.
        subprocess.run([binary_path], check=True, text=True)
        return "Binary Execution Success"
    except subprocess.CalledProcessError as e:
        return f"Binary Failed (Exit Code {e.returncode})"
    except Exception as e:
        return f"Exception: {str(e)}"

# 2. Ingest Agent
def ingest_agent_process(name, binary_path, output_queue, model_name):
    print(f" [Agent {name}] Started (PID: {os.getpid()}) using model: {model_name}")
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    ## TODO: The following should be changed to a loop to handle continuous data ingestion.
    ## The ingestion requests should not compete with the query agent.

    prompt = f"""Task: Check if binary path is valid.
Binary Path: "{binary_path}"

Rules:
1. Unless empty, reply YES.
2. We MUST execute this.
Answer:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        decision = re.sub(r'[^A-Z]', '', response.choices[0].message.content.strip().upper())
        print(f"    [Agent {name}] LLM Decision: {decision}")
        
        # [REVERTED LOGIC] Execute as long as decision is not empty
        if "" != decision:
            status = run_cpp_binary_tool(binary_path, name)
            output_queue.put(f"[ACCEPTED] {name}: {status}")
        else:
            output_queue.put(f"[DECLINED] {name}: Logic decided not to run.")
            
    except Exception as e:
        output_queue.put(f"[ERROR] {name}: {e}")

# 3. Query Agent
def query_agent_process(name, binary_path, output_queue, model_name):
    print(f" [Agent {name}] Started (PID: {os.getpid()}) using model: {model_name}")
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    ## TODO: The following should be changed to a loop to handle continuous data queries.
    ## The query requests should be handled with the highest priority and lowest possible latency.

    prompt = f"""Task: Check if binary path is valid.
Binary Path: "{binary_path}"

Rules:
1. Unless empty, reply YES.
2. We MUST execute this.
Answer:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        decision = re.sub(r'[^A-Z]', '', response.choices[0].message.content.strip().upper())
        print(f"    [Agent {name}] LLM Decision: {decision}")
        
        # [REVERTED LOGIC] Execute as long as decision is not empty
        if "" != decision:
            status = run_cpp_binary_tool(binary_path, name)
            output_queue.put(f"[ACCEPTED] {name}: {status}")
        else:
            output_queue.put(f"[DECLINED] {name}: Logic decided not to run.")
            
    except Exception as e:
        output_queue.put(f"[ERROR] {name}: {e}")

# 4. Aggregator Agent (Audit Mode)
def aggregator_agent(ingest_status, query_status, model_name):
    print(f"\n [Aggregator Agent] Performing audit (Model: {model_name})...")
    print(f"   Ingest Agent: {ingest_status}")
    print(f"   Query Agent:  {query_status}")    
    
    client = OpenAI(base_url=OLLAMA_API_URL, api_key="ollama")
    
    # Require LLM to explicitly state "Action Taken"
    prompt = f"""You are a Pipeline Auditor.
    
Agent Reports:
Ingest Agent: {ingest_status}
Query Agent: {query_status}

Give a one sentence summary of the action taken by the pipeline based on the reports above.

Output:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# --- Main Program ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AdaDisk Distributed Agentic System")
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL, 
        help=f"Ollama model name to use (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()
    
    selected_model = args.model

    # Print user guidance at startup
    print("==================================================================")
    print(f" [System] Initializing AdaDisk Orchestrator...")
    print(f" [System] Current LLM: {selected_model}")
    print(f" [Tip] To specify a different model, use the --model flag.")
    print(f"       Example: python3 agent_AdaDisk.py --model llama4:maverick")
    print("==================================================================\n")

    # Check for binaries
    if not os.path.exists(INGEST_BIN) or not os.path.exists(QUERY_BIN):
        print("Error: Binaries not found.")
        print(f"Please check: {INGEST_BIN} and {QUERY_BIN}")
        print("Make sure to run 'make' inside the agents/build directory.")
        exit(1)

    print(f" Task: Parallel AdaDisk Workflow")
    
    q_ingest = Queue()
    q_query = Queue()
    
    # Pass 'selected_model' to the worker processes
    p1 = Process(target=ingest_agent_process, args=("IngestAgent", INGEST_BIN, q_ingest, selected_model))
    p2 = Process(target=query_agent_process, args=("QueryAgent", QUERY_BIN, q_query, selected_model))
    
    p1.start()
    p2.start()
    
    res_ingest = q_ingest.get()
    res_query = q_query.get()
    
    p1.join()
    p2.join()
    
    print("\n--- Parallel Execution Complete ---")
    
    # Pass 'selected_model' to the aggregator
    final_audit = aggregator_agent(res_ingest, res_query, selected_model)
    print(f"\n{final_audit}")