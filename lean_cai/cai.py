import os
import openai
import subprocess
from model import GPT
from typing import Literal

def verify(theorem: str):
  raise NotImplementedError

class PreferenceData:
    prompt: str
    response1: str
    response2: str
    preference: Literal[0, 1]

def compare(prompt, response1, lean_output1, response2, lean_output2) -> PreferenceData:
    constitution = [
        "Choose the proof that is correct. A proof that doesn't work should NEVER be preferred over another proof that is not correct.",
        "When both proofs are correct/incorrect, choose the proof that is simpler and cleaner.",
        "When both proofs are correct/incorrect, choose the proof with the fewest steps and least dependencies.",
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

    raise NotImplementedError

def revise(prompt, response, critique_output, verification_output) -> str:
    raise NotImplementedError

def critique(prompt, response) -> str:
    raise NotImplementedError

def critique_response(prompt, response) -> str:
    raise NotImplementedError
