# Day 5 Analysis — Serving Models Your Way

## 1. Scenario

My scenario is a **Placement Interview Assistant** for an IT student. I want the model to answer placement-preparation questions in a concise, professional and easy-to-understand way. It should explain technical concepts simply, give practical examples when useful, and avoid unnecessary long answers. This scenario is small enough to test locally but is also realistic enough to show why a local model and a multi-user production server have different requirements.

## 2. What is Ollama?

Ollama is a local model-serving platform that makes it simple to download, manage, configure and run language models on a personal computer. In this task I use Ollama to serve `qwen2.5:1.5b` and my custom `placement-assistant` model.

For this task, Ollama can be understood through three main parts. First is the **model/runtime layer**, which loads the model weights and performs inference. Second is the **Ollama server**, which listens locally, normally at `http://localhost:11434`, and accepts requests from programs. Third is the **client interfaces**, including the command line interface and HTTP APIs. These interfaces allow the same server to be used interactively or from Python applications.

The model data is managed in Ollama's local model storage. The exact physical directory depends on the operating system and Ollama configuration, so the important point for this experiment is that the model is stored locally and reused rather than downloaded for every request.

When I send a request, my CLI or Python program sends it to the Ollama server. Ollama identifies the requested model, loads it into memory if it is not already loaded, runs inference, and returns the generated text. With streaming enabled, generated pieces are returned progressively rather than waiting for the complete answer.

## 3. What is a Modelfile?

A Modelfile is a configuration file used by Ollama to create a customized model definition. In my scenario, I used:

```text
FROM qwen2.5:1.5b
PARAMETER temperature 0.2
PARAMETER num_ctx 2048
SYSTEM ...
```

`FROM qwen2.5:1.5b` selects the already available base model. I selected `temperature 0.2` because a placement assistant should produce stable and focused answers rather than highly random answers. I selected `num_ctx 2048` because the prompts in this small experiment do not require a large context and a smaller context reduces the amount of KV-cache memory needed per request.

The `SYSTEM` instruction defines the default behaviour of the model. It tells the model that it is a Placement Interview Assistant and gives rules about clarity, conciseness, professional tone and examples.

A custom Ollama model does not necessarily mean that a completely new set of neural-network weights has been trained. In this experiment, the custom model is based on the existing Qwen model layers plus the configuration and prompt that define its behaviour. When I ran `ollama create`, Ollama reported that existing layers were being reused. Therefore creating `placement-assistant` did not download another copy of the model weights.

## 4. Which system prompt wins?

The Modelfile provides a default system behaviour. A program can supply its own system message when making a request. The program-level instruction is the more specific instruction for that request and can override the default behaviour provided by the Modelfile.

This is useful for an agent because the developer can package a sensible default behaviour into the model but still allow the application to specialize the model for a particular task, user or workflow. In my experiment, the Python OpenAI-compatible client supplies a different system instruction asking the model to behave as a travel assistant. The result demonstrates that the application can customize the model at request time instead of being permanently restricted to the Modelfile's default role.

## 5. `/api/generate` vs `/api/chat`

`/api/generate` is a generation-oriented endpoint. It accepts a prompt and generates a completion. It is convenient for a simple single-prompt experiment.

`/api/chat` is designed around chat messages and conversation roles such as `user` and `assistant`. It is more natural for applications that maintain conversational history or construct messages dynamically.

Ollama also provides an OpenAI-compatible endpoint under `/v1`. This matters because many Python libraries and AI applications already use the OpenAI client interface. If the application is written against a compatible API, moving from a local Ollama server to another compatible inference server can require much less application-level rewriting. The model-serving backend can change while the application keeps the same basic client pattern.

## 6. Streaming, TTFT and total time

Streaming means that the server sends generated output incrementally as tokens are produced instead of waiting for the entire response before sending anything.

**TTFT (Time To First Token)** measures the time from sending the request until the first generated token becomes available. **Total time** measures the full time required to finish the request.

Streaming does not magically reduce the total computation required to generate the answer. Its main user-experience advantage is that the first text appears earlier. Therefore a user can start reading while the model is still generating the remaining answer, making the system feel faster.

The Python script in this project measures TTFT for a streaming request and records total generation time and generated-token speed using the timing information returned by Ollama.

## 7. `num_ctx`, `OLLAMA_KEEP_ALIVE` and `OLLAMA_NUM_PARALLEL`

`num_ctx` controls the context size used for an inference request. A larger context allows more tokens from the prompt and conversation to be considered, but it also increases memory requirements.

`OLLAMA_KEEP_ALIVE` controls how long a model stays loaded in memory after a request. Keeping a model loaded can reduce the load overhead for a subsequent request, while unloading it can release memory.

`OLLAMA_NUM_PARALLEL` controls how many parallel requests Ollama can process for a loaded model. Increasing parallelism can improve utilization and throughput when there are multiple requests, but it also increases memory requirements.

The KV cache stores attention-related information for tokens already processed during generation. Therefore a larger `num_ctx` can require more KV-cache memory. More simultaneous requests also require more KV-cache space. This is one reason that serving many users is fundamentally a memory-management problem as well as a compute problem.

## 8. Why simple serving wastes GPU memory and how PagedAttention helps

During inference, the KV cache can consume a significant amount of GPU memory. A simple allocation strategy may reserve large contiguous regions for individual sequences even when those regions are not completely used. With many different requests and different sequence lengths, this can create fragmentation and inefficient memory utilization.

PagedAttention, used by vLLM, manages the KV cache using fixed-size blocks. Instead of requiring each request to own one large contiguous cache region, the cache can be divided into blocks. A block table maps logical sequence positions to physical memory blocks. This allows the system to allocate and reuse memory more efficiently.

An additional benefit is prefix caching. If many agent requests begin with the same long system prompt or other shared prefix, the already-computed prefix information can be reused rather than recomputed for every request. This is especially useful for agents because the same system instructions, tool descriptions or other context may be sent repeatedly.

## 9. Static batching vs continuous batching

Static batching groups a known set of requests together and processes them as a batch. It can improve hardware utilization, but it may have to wait for the batch to be formed and may be inefficient when requests arrive at different times or have different lengths.

Continuous batching dynamically adds and removes requests while the server is running. As one request finishes or reaches a suitable point, another request can be admitted. This allows the server to keep the hardware busy while handling a changing stream of requests.

**Throughput** is the amount of useful work completed per unit time, such as generated tokens per second across users. **TTFT** is time to first token. **TPOT (Time Per Output Token)** measures the average time associated with producing output tokens after generation begins. **P95 latency** is the latency value below which 95% of requests finish; it is useful because it shows tail behaviour rather than only the average.

Increasing concurrency or batching can improve throughput, but it can also increase waiting and memory pressure. That can increase TTFT or P95 latency. Therefore a production server has to balance throughput, responsiveness and memory rather than maximizing only one metric.

## 10. Ollama vs vLLM comparison

| Basis | Ollama | vLLM |
|---|---|---|
| Built for | Simple local model running, experimentation and individual developers; excellent for a small number of users | High-throughput serving and production workloads with many concurrent requests |
| Hardware it needs | Works well on a developer machine with CPU or a suitable local GPU; model size must fit available resources | Commonly deployed with capable GPUs and enough VRAM for production inference; optimized for high-performance serving |
| Several requests at once | Supports parallel requests, but is primarily focused on convenient local serving | Designed around efficient concurrent serving, continuous batching and high-throughput scheduling |
| Memory management | Straightforward model loading and runtime management; KV cache still consumes memory per active request | Advanced KV-cache management with PagedAttention and block-based allocation |
| Setup effort and model format | Very easy local setup; Modelfile makes customization simple | More production-oriented deployment and configuration; supports efficient serving of supported model formats |
| My choice if only I use the scenario | **Ollama**, because one user needs simplicity, local execution and easy model customization | vLLM would be unnecessary complexity for this small personal workload |
| My choice if 100 people use it at once | **Not my first choice** for this requirement because concurrency, scheduling and memory efficiency become important | **vLLM**, because it is designed for high-concurrency serving, continuous batching and efficient KV-cache management |

## 11. Observations

The initial model creation reused existing Qwen layers. Ollama therefore did not need to download a second copy of the 986 MB model weights. The `ollama show placement-assistant` output confirmed the Qwen2 architecture, 1.5B parameter size, Q4_K_M quantization, `num_ctx 2048`, `temperature 0.2`, and the Placement Interview Assistant system prompt.

The first request can include model-loading overhead if the model was not already resident in memory. A subsequent request while the model remains loaded should normally avoid that initial load cost. The `ollama ps` command is used during the experiment to verify whether `placement-assistant` is loaded.

The exact TTFT, total time and tokens-per-second values are intentionally generated by `ollama_test.py` on the local machine rather than hard-coded in this report. This makes the measurements reproducible and avoids presenting fabricated benchmark numbers. The script performs both non-streaming and streaming requests and prints the measured values.

### Observation table to complete from the real run

| Run | Prompt | Behaviour followed? | TTFT | Total time | Tokens/sec | Model loaded? |
|---|---|---|---:|---:|---:|---|
| 1 | Explain DevOps in simple terms for a placement interview | Record actual result | 6.959 s | 6.959 s | 85.64 | Yes - `ollama ps` showed 100% GPU and 2048 context |
| 2 | What is Docker? Give a simple explanation and example | Record actual result | 2.171 s | 3.253 s | 83.93 | Yes - model remained available during the test |
| 3 | Give a long unnecessary explanation of Kubernetes | Record actual result | Measured in the local run | Measured in the local run | Measured in the local run | Check `ollama ps` |
| 4 | Repeat the first prompt | Record actual result | Measured in the local run | Measured in the local run | Measured in the local run | Check `ollama ps` |

The repeated run should be compared with the first run. If the model remained loaded, the second request can avoid the initial model-loading overhead. Streaming should make the response feel faster because the first generated text becomes visible before the complete response has finished, even when the total generation time is similar.

## 12. Suitability and conclusion

For my Placement Interview Assistant, Ollama on a single machine is good enough when the main requirement is personal experimentation, development, testing prompts, building a prototype, or serving a small number of users. The command-line interface makes interactive testing easy, the Modelfile packages the default behaviour, and the REST API makes the model available to Python applications.

The Modelfile adds reproducible configuration. Instead of manually sending the same role and behavioural instructions every time, I can create `placement-assistant` with a fixed default behaviour. The REST API then turns the local model into a service that programs can call. The OpenAI-compatible endpoint makes the application easier to move between compatible model servers.

The situation changes when many users send requests at the same time. A shared deployment has to manage GPU memory, KV caches, request scheduling, batching, latency and throughput. For a workload such as 100 simultaneous users, a production-oriented server such as vLLM is a better fit because PagedAttention improves KV-cache memory management and continuous batching keeps the accelerator better utilized while requests arrive and finish at different times.

Overall, Ollama is a strong choice for local development and small-scale serving, while vLLM is more appropriate when the problem becomes a high-concurrency model-serving problem. The important lesson is that choosing a model is only part of an AI system: the serving layer determines how efficiently that model can be used by one person or by many users.
