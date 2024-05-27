import json
import random
from typing import List, Dict
from pydantic import BaseModel
from model import GPT

class PreferenceExample(BaseModel):
    prompt: str
    chosen: str
    rejected: str

def evaluate_gpt(examples: List[PreferenceExample], gpt: GPT) -> Dict[str, float]:
    correct = 0
    total = len(examples)
    c = 0

    for example in examples:
      # add randomness to the order of the two completions
        correct_num = random.randint(1, 2)
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps with coding tasks."},
            {"role": "user", "content": f"Given the following prompt:\n{example.prompt}\n\nWhich of the following two completions is better?\n\nCompletion 1:\n{example.rejected if correct_num == 1 else example.chosen}\n\nCompletion 2:\n{example.chosen if correct_num == 1 else example.rejected}\n\nPlease provide only the number of the better completion (1 or 2). respond ONLY with the number of the better completion"}
        ]
        response = gpt.generate(messages, None)

        print(response)

        if response.strip() == str(correct_num):
            correct += 1
            c += 1
        elif response.strip() == "1" or response.strip() == "2":
            c += 1

        print(correct / c)

    accuracy = correct / total
    return {"accuracy": accuracy}

def main():
    with open("./eval_final.jsonl", "r") as file:
        examples = [PreferenceExample(**json.loads(line)) for line in file]

    examples = examples[-300:]

    gpt = GPT(api_key="0cb4bc731905be49d7fd383487bca1f51e8a05dd955c29318e53b4f29559d7cd")
    results = evaluate_gpt(examples, gpt)
    print(f"Accuracy: {results['accuracy']:.2%}")

if __name__ == "__main__":
    main()