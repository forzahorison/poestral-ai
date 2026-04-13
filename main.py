import fastapi_poe as fp
from openai import AsyncOpenAI
import os
import json
import sys

class MistralBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=os.environ["MISTRAL_API_KEY"],
            base_url="https://api.mistral.ai/v1",
        )

    async def get_response(self, request: fp.QueryRequest):
        messages = []

        messages.append({
            "role": "system",
            "content": "You are a helpful assistant."  # ganti sesuai kebutuhan
        })

        for msg in request.query:
            role = msg.role
            if role == "bot":
                role = "assistant"
            messages.append({"role": role, "content": msg.content})

        # Batasi history
        MAX_MESSAGES = 10
        if len(messages) > MAX_MESSAGES + 1:
            messages = [messages[0]] + messages[-MAX_MESSAGES:]

        print(f"Messages count: {len(messages)}", file=sys.stderr)
        print(f"Payload size: {len(json.dumps(messages))} bytes", file=sys.stderr)

        stream = await self.client.chat.completions.create(
            model="mistral-large-2411",
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield fp.PartialResponse(text=delta)

app = fp.make_app(MistralBot(), access_key=os.environ["POE_ACCESS_KEY"])
