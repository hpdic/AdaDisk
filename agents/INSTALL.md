# System Setup
> Dec. 11, 2025, Ubuntu 24.04, CPU 4 sockets 224 cores, RAM 2.9TB

## Install Ollama
```bash
sudo apt update
git config --global user.name "Dongfang Zhao"
git config --global user.email "dongfang.zhao@gmail.com"
sudo apt install python3-pip -y
curl -fsSL https://ollama.com/install.sh | sh
sudo update-alternatives --config editor
```

## Keep Ollama model in memory
```bash
sudo systemctl edit ollama.service
```
Add the following lines:
```bash
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama | grep Environment
```

## Start using Ollama
```bash
ollama list
ollama pull llama3.2:1b
ollama run llama4:maverick # 244GB
ollama stop qwen3:235b
ollama rm qwen3:235b
```

## API Access
```bash
cd ~/Kaimosia
python3 -m venv venv
source venv/bin/activate
pip install ollama
cd examples
python helloworld.py
``` 

## Agentic AI with Ollama LLMs
```bash
pip install -U langchain langgraph langchain-ollama openai 
cd examples
python hello_agent.py
python hello_langchain.py
python hello_langgraph.py
python hello_parallel_agents.py
```

## Update the default Ollama data directory
By default, Ollama stores models in `/var/lib/ollama/models`, which may have limited space. To check the current data directory:
```bash
systemctl show ollama | grep OLLAMA_MODELS
```

If you need more disk space, you can change the Ollama data directory:
```bash
sudo systemctl edit ollama.service
```
Add the following lines:
```bash
[Service]
Environment="OLLAMA_MODELS=/dev/shm/ollama_models"
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama | grep OLLAMA_MODELS
```
If you want to move existing models to the new directory, you can do so with:
```bash
sudo cp -rp /usr/share/ollama/.ollama/models/* /dev/shm/ollama_models/
```