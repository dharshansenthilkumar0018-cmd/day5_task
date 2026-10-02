import json
import time
import requests

BASE_URL = "http://localhost:11434"
MODEL = "placement-assistant"


def non_streaming(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 2048},
    }

    start = time.perf_counter()
    response = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    response.raise_for_status()
    end = time.perf_counter()

    data = response.json()
    total = end - start
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)

    tokens_per_sec = (
        eval_count / (eval_duration_ns / 1e9)
        if eval_count and eval_duration_ns
        else 0
    )

    print("\n=== NON-STREAMING /api/generate ===")
    print("Prompt:", prompt)
    print("Response:", data.get("response", "").strip())
    print(f"Total time: {total:.3f} s")
    print(f"Generated tokens: {eval_count}")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")

    return data


def streaming(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 2048},
    }

    start = time.perf_counter()
    first_token_time = None
    chunks = []
    final_data = {}

    with requests.post(
        f"{BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            item = json.loads(line.decode("utf-8"))
            token = item.get("response", "")

            if token and first_token_time is None:
                first_token_time = time.perf_counter()

            if token:
                print(token, end="", flush=True)
                chunks.append(token)

            if item.get("done"):
                final_data = item

    end = time.perf_counter()

    ttft = (first_token_time - start) if first_token_time else 0
    total = end - start
    eval_count = final_data.get("eval_count", 0)
    eval_duration_ns = final_data.get("eval_duration", 0)

    tokens_per_sec = (
        eval_count / (eval_duration_ns / 1e9)
        if eval_count and eval_duration_ns
        else 0
    )

    print("\n\n=== STREAMING /api/generate ===")
    print(f"TTFT: {ttft:.3f} s")
    print(f"Total time: {total:.3f} s")
    print(f"Generated tokens: {eval_count}")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")

    return final_data, ttft, total


def chat_api():
    messages = [
        {
            "role": "user",
            "content": "What is CI/CD? Explain it for a placement interview.",
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    print("\n=== /api/chat ===")
    print(data.get("message", {}).get("content", "").strip())
    return data


if __name__ == "__main__":
    prompt = "Explain DevOps in simple terms for a placement interview."

    non_streaming(prompt)
    print("\nStreaming response:")
    streaming(prompt)
    chat_api()
