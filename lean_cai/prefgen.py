import concurrent.futures
import pickle
from typing import List
from pydantic import BaseModel
from lean import LeanProofChecker
from model import Claude
from cai import CAIQuestionData, Round

def revise(prompt, response, critique_output, verification_output, prev_rounds: List[Round], model, system_prompt=None) -> str:
    revise_system_prompt = "You are a helpful assistant helping with mathematical formalization in Lean, and you revise a response based on the following critique and Lean compiler output. Put all code in code blocks that are of the format ```lean\n<code>\n```"
    revise_system_prompt += "\n\n"
    if len(prev_rounds) > 0:
        revise_system_prompt += "For further context on revisions, here were the prior rounds of revisions:"
        for round in prev_rounds:
            revise_system_prompt += f"""Round {prev_rounds.index(round) + 1}: \n Critique: {round.critique_output}\n Verification: {round.verification_output}\n Revision: {round.revise_output}\n"""
    revise_system_prompt += "Reminder: All of your lean code should be in a code block of the format ```lean\n<code>\n```"
    user_prompt = f"""This round (round to revise): \n Prompt: {prompt}\n Response: {response}\n Critique: {critique_output}\n Verification: {verification_output}"""
    response = model.generate(user_prompt, system_prompt=revise_system_prompt if system_prompt is None else system_prompt)
    return response

def critique(prompt, response, verification_output, model, system_prompt=None) -> str:
    critic_system_prompt = "You are a helpful assistant helping with mathematical formalization in Lean, and you critique a response based on the following prompt and Lean compiler output. \n\n You also have the below constitution that you must follow when critiquing:\n"
    constitution = [
        "Correctness is the most important criterion when critiquing a proof.",
        "Incomplete proofs (when `sorry` is in the verification output) should not exist in responses, and a critique should always be made in this case",
        "Proof should be simple/clean, though never at the expense of correctness",
        "Proof should be as short as possible, though never at the expense of being simple/clean",
    ]
    for rule in constitution:
        critic_system_prompt += rule + "\n"
    user_prompt = f"""Prompt: {prompt} Response: {response} Verification: {verification_output}"""
    response = model.generate(user_prompt, system_prompt=critic_system_prompt if system_prompt is None else system_prompt)
    return response

# if __name__ == "__main__":
#     dataset = pickle.load(open("./cai_data_no_round.pkl", "rb"))
#     model = Claude(api_key="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")
#     verifier = LeanProofChecker()
#     for i in range(len(dataset)):
#         print("iteration", i)
#         response = dataset[i].response
#         for r in range(5):
#             verification_output = verifier.verify(response)
#             critique_output = critique(dataset[i].prompt, dataset[i].response, verification_output, model)
#             revise_output = revise(dataset[i].prompt, dataset[i].response, critique_output, verification_output, dataset[i].rounds, model)
#             try:
#                 revise_output = revise_output.split("```lean\n")[1].split("\n```")[0]
#             except:
#                 pass
#             dataset[i].rounds.append(Round(verification_output=verification_output, critique_output=critique_output, revise_output=revise_output))
#             response = revise_output
#             print("round", r)
#             # print(verification_output)
#             # print(critique_output)
#             # print(revise_output)
#         pickle.dump(dataset, open(f"./data/cai/cai_data_round_{i}.pkl", "wb"))
#     pickle.dump(dataset, open("./cai_data_with_rounds.pkl", "wb"))


class Round:
    def __init__(self, verification_output, critique_output, revise_output):
        self.verification_output = verification_output
        self.critique_output = critique_output
        self.revise_output = revise_output

def process_item(item, model, verifier):
    response = item.response
    for r in range(5):
        verification_output = verifier.verify(response)
        critique_output = critique(item.prompt, item.response, verification_output, model)
        revise_output = revise(item.prompt, item.response, critique_output, verification_output, item.rounds, model)
        try:
            revise_output = revise_output.split("```lean\n")[1].split("\n```")[0]
        except:
            pass
        item.rounds.append(Round(verification_output=verification_output, critique_output=critique_output, revise_output=revise_output))
        response = revise_output
        print(f"Round {r} for item {item}")
    return item

if __name__ == "__main__":
    dataset = pickle.load(open("./cai_data_no_round.pkl", "rb"))[22:]
    model = Claude(api_key="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")
    verifier = LeanProofChecker()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(len(dataset)):
            future = executor.submit(process_item, dataset[i], model, verifier)
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            dataset[i] = future.result()
            pickle.dump(dataset, open(f"./data/cai/cai_data_round_{i}.pkl", "wb"))

    pickle.dump(dataset, open("./cai_data_with_rounds.pkl", "wb"))