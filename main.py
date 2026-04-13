import fastapi_poe as fp
from openai import AsyncOpenAI, RateLimitError
import asyncio
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
            "content": "You are a helpful assistant."
        })

        for msg in request.query:
            role = msg.role
            if role == "bot":
                role = "assistant"
            messages.append({"role": role, "content": msg.content})

        MAX_MESSAGES = 10
        if len(messages) > MAX_MESSAGES + 1:
            messages = [messages[0]] + messages[-MAX_MESSAGES:]

        print(f"Messages count: {len(messages)}", file=sys.stderr)
        print(f"Payload size: {len(json.dumps(messages))} bytes", file=sys.stderr)

        MAX_RETRIES = 3
        RETRY_DELAY = 5  # detik

        for attempt in range(MAX_RETRIES):
            try:
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
                return  # sukses, keluar dari loop

            except RateLimitError as e:
                print(f"Rate limit hit (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    yield fp.PartialResponse(text=f"⏳ Server busy, retrying in {RETRY_DELAY}s...\n")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    yield fp.PartialResponse(text="❌ Server sedang overloaded, coba lagi beberapa saat.")

app = fp.make_app(MistralBot(), access_key=os.environ["POE_ACCESS_KEY"])
