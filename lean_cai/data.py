from lean_dojo import LeanGitRepo, Theorem, get_traced_repo_path, trace, TracedRepo
from typing import List
import json
import random
import pickle
from tqdm import tqdm
import concurrent.futures
import logging

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

train_data = json.load(open('./leandojo_benchmark_4/random/train.json'))
repos = {}
traced_repos = {}
theorems = []

import os
if os.path.exists('./traced_repo.pkl'):
    traced_repo = pickle.load(open('./traced_repo.pkl', 'rb'))
else:
    repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "fe4454af900584467d21f4fd4fe951d29d9332a7")
    traced_repo = trace(repo)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_theorem(theorem_data):
    url, commit, file_path, full_name, traced_tactics, start, end = theorem_data['url'], theorem_data['commit'], theorem_data['file_path'], theorem_data['full_name'], theorem_data['traced_tactics'], theorem_data['start'], theorem_data['end']
    logging.debug(f"Processing theorem from {url} at commit {commit}, file {file_path}, name {full_name}")

    for traced_file in traced_repo.traced_files:
        if str(traced_file.path) == file_path:
            traced_theorem = traced_file.get_traced_theorem(full_name)
            if traced_theorem:
                logging.debug(f"Found traced theorem {full_name} in file {file_path}")
                if random.random() < 0.1:
                    logging.debug(f"Random selection passed for theorem {full_name}")
                    if len(traced_theorem.get_premise_full_names()) > 0:
                        name = traced_theorem.theorem.full_name
                        statement = traced_theorem.get_theorem_statement()
                        proof = traced_theorem.get_tactic_proof()
                        imports = traced_file.get_direct_dependencies(traced_repo.repo)
                        comments = [str(comment) for comment in traced_theorem.comments]
                        theorem = LeanTheorem(name, statement, proof, imports, comments)
                        logging.info(f"Theorem {name} processed successfully")
                        return theorem
                    else:
                        logging.debug(f"No premises found for theorem {full_name}")
            else:
                logging.debug(f"No traced theorem found for {full_name} in file {file_path}")

    logging.info(f"No suitable theorem found for processing in file {file_path}")
    return None

with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(process_theorem, theorem_data) for theorem_data in train_data]

    with tqdm(total=len(train_data), unit='theorem') as pbar:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                theorems.append(result)
            pbar.update(1)

logging.info(f"Processed {len(theorems)} theorems.")
pickle.dump(theorems, open('./data/theorems.pkl', 'wb'))