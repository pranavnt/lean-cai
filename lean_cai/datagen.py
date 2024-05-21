from pydantic import BaseModel
import logging
import pickle
from tqdm import tqdm
from typing import Literal, List, Dict, Any
from concurrent import futures
from model import GPT



class LeanTheorem:
    name: str
    statement: str
    proof: str
    imports: List[str]
    comments: List[str]

    def __init__(self, name: str, statement: str, proof: str, imports: List[str], comments: List[str]):
        self.name = name
        self.statement = statement
        self.proof = proof
        self.imports = imports
        self.comments = comments

class QuestionModel(BaseModel):
  question: str

class AnswerModel(BaseModel):
  answer: str

def generate_prove_qa(proof: LeanTheorem, model: GPT = None):
    SYSTEM_PROMPT = "You are a helpful assistant helping with mathematical formalization in Lean."

    return [
        {
        "role": "system",
        "content": SYSTEM_PROMPT
        },
        {
        "role": "user",
        "content": f"Can you write a proof for the following theorem?\n```\n{proof.statement}\n```"
        },
        {
        "role": "assistant",
        "content": f"Here's the proof in lean:\n```\n{proof.statement}\n{proof.proof}\n```"
        }
    ]

def generate_statement_qa(proof: LeanTheorem, model: GPT = None):
  SYSTEM_PROMPT = "You are a helpful assistant helping with mathematical formalization in Lean."
  PROMPT = f"""Generate a question that prompts a user to formalize the Lean statement below.

From the Lean statement:
```
theorem length_merge (s : α → α → Bool) (l r) :\n    (merge s l r).length = l.length + r.length :=
```,
Question: "Formalize the statement that the length of the merge of two lists is equal to the sum of the lengths of the two lists in Lean"

What is the question for the below lean statement?
```
{proof.statement}
```
"""
  messages = [
    {
      "role": "system",
      "content": SYSTEM_PROMPT
    },
    {
      "role": "user",
      "content": PROMPT
    }
  ]

  response = model.generate(messages, QuestionModel)

  return [
    {
      "role": "system",
      "content": SYSTEM_PROMPT
    },
    {
      "role": "user",
      "content": response["question"]
    },
    {
      "role": "assistant",
      "content": f"Here's the formalized statement:\n```\n{proof.statement}\n```"
    }
  ]

STRATEGIES = {
  "PROVE": {
    "function": generate_prove_qa,
    "requires": ["statement", "proof"]
  },
  "STATEMENT": {
    "function": generate_statement_qa,
    "requires": ["statement"]
  }
}

def generate_qa(strategies: List[Dict[str, Any]], proof: LeanTheorem, model: GPT = None):
    results = []
    for strategy_name, strategy in strategies.items():
        try:
            if "proof" in strategy["requires"] and (proof.proof is None or proof.proof == ""):
                continue
            else:
                result = strategy["function"](proof, model)
                results.append(result)
        except Exception as e:
            logging.warning(f"Error processing {strategy_name} for theorem {proof.name}: {str(e)}")
    return results

if __name__ == "__main__":
    gpt = GPT(api_key="0cb4bc731905be49d7fd383487bca1f51e8a05dd955c29318e53b4f29559d7cd")
    theorems = pickle.load(open("./data/theorems_f.pkl", "rb"))

    data = []
    progress_bar = tqdm(total=len(theorems[16000:]), desc="Processing theorems")
    save_interval = 1000

    with futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_objects = [executor.submit(generate_qa, STRATEGIES, theorem, gpt) for theorem in theorems[16000:]]
        for future in futures.as_completed(future_objects):
            data.extend(future.result())
            progress_bar.update(1)
            if len(data) % save_interval == 0:
                pickle.dump(data, open(f"./data/qa/qa_data_{len(data)}.pkl", "wb"))

    pickle.dump(data, open("./data/qa_data.pkl", "wb"))
