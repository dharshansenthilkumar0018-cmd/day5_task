# Agentic AI - Day 5

## Serving Models Your Way: Ollama, Modelfiles, REST API and vLLM

### Scenario
Placement Interview Assistant for an IT student.

### Base model
`qwen2.5:1.5b`

### Custom model
`placement-assistant`

### Files
- `Modelfile` - custom behaviour and model parameters
- `ollama_test.py` - REST API tests for non-streaming, streaming and `/api/chat`
- `openai_test.py` - OpenAI-compatible endpoint test
- `analysis.md` - conceptual analysis, comparison, observations and conclusion
- `requirements.txt` - Python dependency
- `screenshots/` - evidence screenshots

## Setup

```powershell
python -m pip install -r requirements.txt
ollama create placement-assistant -f Modelfile
ollama list
ollama show placement-assistant
python ollama_test.py
```

For the OpenAI-compatible script:

```powershell
python -m pip install openai
python openai_test.py
```

## Important
Performance values such as TTFT, total time and tokens/sec must be generated on the machine running Ollama. Do not invent them. `ollama_test.py` prints the real values automatically.
