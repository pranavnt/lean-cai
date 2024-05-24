import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
from pydantic import BaseModel
from typing import List
import openai
import subprocess
from model import GPT, Claude
from typing import Literal
from posttrain_data import get_dataset
from lean import LeanProofChecker

class PreferenceData:
    prompt: str
    response1: str
    response2: str
    preference: Literal[0, 1]

class Round:
    verification_output: str
    critique_output: str
    revise_output: str

    def __init__(self, verification_output: str, critique_output: str, revise_output: str):
        self.verification_output = verification_output
        self.critique_output = critique_output
        self.revise_output = revise_output

class CAIQuestionData:
    prompt: str
    response: str
    rounds: List[Round]

    def __init__(self, prompt: str, response: str, rounds: List[Round]):
        self.prompt = prompt
        self.response = response
        self.rounds = rounds

class AnswerModel(BaseModel):
  answer: str

def compare(prompt, response1, lean_output1, response2, lean_output2, model) -> PreferenceData:
    constitution = [
        "Choose the proof that is correct. A proof that doesn't work should NEVER be preferred over another proof that is not correct.",
        "When both proofs are correct/incorrect, choose the proof that is simpler and cleaner.",
        "When both proofs are correct/incorrect, choose the proof with the fewest steps and least dependencies.",
        "If a proof is incomplete (has `sorry`), it should not be preferred over another proof that is complete or at least has some progress"
    ]

    critic_system_prompt = "You are a helpful assistant helping with mathematical formalization in Lean, and you compare two responses and return the better one based on the following Constitution (and the output of the Lean proof checker):\n"

    for rule in constitution:
        critic_system_prompt += rule + "\n"

    user_prompt = f"""
Prompt: {prompt}
Response 1: {response1}
Lean proof checker output: {lean_output1}
Response 2: {response2}
Lean proof checker output: {lean_output2}"""

    messages = [
        {"role": "system", "content": critic_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    class PreferenceResponse(BaseModel):
        reasoning: str
        preference: Literal[0, 1]

    response = model.generate(messages, PreferenceResponse)

    return PreferenceData(prompt=prompt, response1=response1, response2=response2, preference=response["preference"])


def process_data_point(data_point, claude):
    response = claude.generate(data_point.train_str())
    proof = response.split("```lean\n")[1].split("\n```")[0]
    print(proof)
    return CAIQuestionData(prompt=data_point.train_str(), response=proof, rounds=[])

def init_data_no_round(claude: Claude, dataset):
    with ThreadPoolExecutor() as executor:
        cai_data = list(tqdm(executor.map(lambda data_point: process_data_point(data_point, claude), dataset), total=len(dataset)))

    pickle.dump(cai_data, open("cai_data_no_round.pkl", "wb"))


if __name__ == "__main__":
    # gpt = GPT(api_key="0cb4bc731905be49d7fd383487bca1f51e8a05dd955c29318e53b4f29559d7cd")
    # claude = Claude(api_key="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")

    dataset = pickle.load(open("./cai_data_no_round.pkl", "rb"))

    for index, entry in enumerate(dataset):
        print(index, len(entry.prompt), len(entry.response))

    # dataset = get_dataset()
    # init_data_no_round(claude, dataset)
    # verifier = LeanProofChecker()

    # cai_data = []

    # for data_point in dataset:
    #     print(data_point.train_str())
    #     # get response
    #     response = claude.generate(data_point.train_str())
    #     proof = response.split("```lean\n")[1].split("\n```")[0]

    #     cai_data_point = CAIQuestionData(prompt=data_point.train_str(), response=proof, rounds=[])

    #     for round_number in range(5):
    #         verifier_output = verifier.verify(proof)

    #         critique_output = critique(data_point.train_str(), proof, verifier_output, gpt)

    #         response = revise(data_point.train_str(), proof, critique_output, verifier_output, cai_data_point.rounds, gpt)
    #         proof = response

    #         cai_data_point.rounds.append(Round(verifier_output, critique_output, response))

    #         print("Round " + str(round_number) + " complete")
    #         print("Proof: " + proof)
    #         print("Critique: " + critique_output)
    #         print("Verifier: " + verifier_output)
    #     pickle.dump(cai_data_point, open("cai_data.pkl", "wb"))
    #     exit()
    #     cai_data.append(cai_data_point)