from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./llama3-mathlib-sft"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def sample(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 800):
  messages_arr = [
    {"role": "system", "content": "You are a helpful assistant helping with mathematical formalization in Lean."},
    {"role": "user", "content": prompt}
  ]

  messages = "<|begin_of_text|>"

  for message in messages_arr:
    role = message['role']
    content = message['content'].replace('\n', '\\n')

    messages += f"<|start_header_id|>{role}<|end_header_id|>\n"
    messages += f"{content}<|eot_id|>"

    messages += "<|start_header_id|>assistant<|end_header_id|>\n"

  input_ids = tokenizer.encode(messages, return_tensors="pt")
  output = model.generate(input_ids, max_new_tokens=max_new_tokens)
  return tokenizer.decode(output[0])

print(sample(model, tokenizer, """Prove the following theorem:

theorem aime_1983_p1
  (x y z w : ℕ)
  (ht : 1 < x ∧ 1 < y ∧ 1 < z)
  (hw : 0 ≤ w)
  (h0 : real.log w / real.log x = 24)
  (h1 : real.log w / real.log y = 40)
  (h2 : real.log w / real.log (x * y * z) = 12):
  real.log w / real.log z = 60 :=
begin
"""))


constitution = [
  ""
]