import os
import pickle
os.environ["GITHUB_ACCESS_TOKEN"]="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"

from lean_dojo import LeanGitRepo, Theorem, get_traced_repo_path, trace, TracedRepo

mil_textbook = {
  'url': 'https://github.com/ImperialCollegeLondon/formalising-mathematics-2024',
  'commit': '2c4b855739396253fb77f23ab43079c43a2746ac'
}

mil_repo = LeanGitRepo(mil_textbook['url'], mil_textbook['commit'])

traced_mil_repo = trace(mil_repo, build_deps=False)

pickle.dump(traced_mil_repo, open("traced_mil_repo.pkl", "wb"))
