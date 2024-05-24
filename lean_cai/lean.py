import json
import os
import shutil
import subprocess
from posttrain_data import get_dataset
from typing import List

class LeanProofChecker:
    def __init__(self):
        self.name = "lean_proof_checker"
        self.project_dir = "./" + self.name
        subprocess.run(["lake", "new", self.name, "math"])

    def verify(self, code: str) -> (str):
        with open(os.path.join(self.project_dir, "LeanProofChecker", "Basic.lean"), "w") as f:
            f.write(code)

        try:
            result = subprocess.run(["lake", "build"], cwd=self.project_dir, capture_output=True, text=True)
            stdout = result.stdout
            stdout = [line for line in stdout.split("\n") if (not "Building" in line and not line.endswith("...") and "Computing build jobs" not in line and "proofwidgets cloud release" not in line)]
            stdout = "\n".join(stdout)
            return stdout
        except FileNotFoundError:
            return False, "Lean executable not found. Please make sure Lean is installed and accessible."
        except Exception as e:
            return False, f"An unexpected error occurred: {str(e)}"

    def cleanup(self):
        shutil.rmtree(self.project_dir)

if __name__ == "__main__":
    dataset = get_dataset()
    checker = LeanProofChecker()
    for d in dataset:
        print(checker.verify(d.theorem))
    checker.cleanup()
