from openai import OpenAI

# Ollama exposes an OpenAI-compatible API at this base URL.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="placement-assistant",
    messages=[
        {
            "role": "system",
            "content": (
                "Ignore the Modelfile's placement-assistant role. "
                "Act instead as a friendly travel assistant. "
                "Answer briefly."
            ),
        },
        {
            "role": "user",
            "content": "Give me one short recommendation for a weekend trip.",
        },
    ],
    temperature=0.2,
)

print("=== OPENAI-COMPATIBLE ENDPOINT ===")
print(response.choices[0].message.content)
