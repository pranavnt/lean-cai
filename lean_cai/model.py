from openai import OpenAI
import os
from anthropic import Anthropic
import json

class GPT:
    def __init__(self, params={}, api_key=None):
        self.params = params
        self.client = OpenAI(api_key=api_key, base_url="https://api.together.xyz")

    def generate(self, messages, data_model):
        max_tokens = self.params.get("max_tokens", 20000)
        model = "mistralai/Mixtral-8x7b-Instruct-v0.1"

        if data_model is not None:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.params.get("temperature", 0.2),
                response_format={
                    "type": "json_object",
                    "schema": data_model.model_json_schema(),
                },
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.params.get("temperature", 0.2),
                frequency_penalty=1.0
            )

            try:
                json.loads(response.choices[0].message.content)
            except:
                print("Failed to parse response as JSON")
                print(response)
                exit()

        return json.loads(response.choices[0].message.content)

class Claude:
    def __init__(self, params={}, api_key=None):
        self.params = params
        self.client = Anthropic(api_key=api_key)

    def generate(self, message: str) -> str:
        max_tokens = self.params.get("max_tokens", 2048)
        model = "claude-3-sonnet-20240229"

        response = self.client.messages.create(
            model=model,
            system="You are a helpful assistant helping with mathematical formalization in Lean. Keep your proofs short, and it's okay (though strongly discouraged) to give up if your responses are too long. Your proofs must be in Lean4 code and that code must be conatined in a ```lean\n``` code block.",
            messages=[
                {"role": "user", "content": message}
            ],
            max_tokens=max_tokens
        )

        return response.content[0].text

from anthropic import Anthropic

class Claude:
    def __init__(self, params={}, api_key=None):
        self.params = params
        self.client = Anthropic(api_key=api_key)

    def generate(self, user_prompt: str, system_prompt: str = None) -> str:
        max_tokens = self.params.get("max_tokens", 2048)
        model = "claude-3-sonnet-20240229"
        if system_prompt is not None:
            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
            )
        else:
            response = self.client.messages.create(
                model=model,
                system="You are a helpful assistant helping with mathematical formalization in Lean. Keep your proofs short, and it's okay (though strongly discouraged) to give up if your responses are too long. Your proofs must be in Lean4 code and that code must be conatined in a ```lean\n``` code block.",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens
            )
        return response.content[0].text

if __name__ == "__main__":
    claude = Claude(api_key="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")
    response = claude.generate("Can you write a proof for the following theorem?\n\n 1+1=2")
    print(response)
