from openai import OpenAI
import json

class GPT:
    def __init__(self, params={}, api_key=None):
        self.params = params
        self.client = OpenAI(api_key=api_key, base_url="https://api.together.xyz")

    def generate(self, messages, data_model):
        max_tokens = self.params.get("max_tokens", 2048)
        model = "mistralai/Mistral-7b-Instruct-v0.1"

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.params.get("temperature", 0),
            response_format={
                "type": "json_object",
                "schema": data_model.model_json_schema(),
            },
        )

        return json.loads(response.choices[0].message.content)