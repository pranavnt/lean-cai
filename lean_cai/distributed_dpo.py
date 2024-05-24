import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import json
import wandb
import torch.distributed as dist
import os

class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer):
        with open(jsonl_path, "r") as file:
            self.data = [json.loads(line) for line in file]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]

        chosen = self.tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item["chosen"]}
        ], tokenize=False)
        rejected = self.tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item["rejected"]}
        ], tokenize=False)

        preferred_tokens = self.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True)
        dispreferred_tokens = self.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True)

        return {
            "preferred_input_ids": preferred_tokens.input_ids.squeeze(),
            "dispreferred_input_ids": dispreferred_tokens.input_ids.squeeze(),
        }

def preference_loss(policy_preferred_logps, policy_dispreferred_logps, reference_preferred_logps, reference_dispreferred_logps, beta):
    pi_logratios = policy_preferred_logps - policy_dispreferred_logps
    ref_logratios = reference_preferred_logps - reference_dispreferred_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    return losses.mean()

def collate_fn(batch):
    preferred_input_ids = torch.nn.utils.rnn.pad_sequence([item["preferred_input_ids"] for item in batch], batch_first=True, padding_value=0)
    dispreferred_input_ids = torch.nn.utils.rnn.pad_sequence([item["dispreferred_input_ids"] for item in batch], batch_first=True, padding_value=0)
    return {"preferred_input_ids": preferred_input_ids, "dispreferred_input_ids": dispreferred_input_ids}

def train(policy_model, reference_model, dataloader, optimizer, beta, epoch):
    policy_model.train()
    reference_model.eval()

    total_loss = 0
    total_policy_preferred_loss = 0
    total_policy_dispreferred_loss = 0
    total_reference_preferred_loss = 0
    total_reference_dispreferred_loss = 0

    for batch in dataloader:
        preferred_input_ids = batch["preferred_input_ids"].to(policy_model.device)
        dispreferred_input_ids = batch["dispreferred_input_ids"].to(policy_model.device)

        policy_preferred_logits = policy_model(preferred_input_ids, labels=preferred_input_ids).logits
        policy_dispreferred_logits = policy_model(dispreferred_input_ids, labels=dispreferred_input_ids).logits
        reference_preferred_logits = reference_model(preferred_input_ids, labels=preferred_input_ids).logits
        reference_dispreferred_logits = reference_model(dispreferred_input_ids, labels=dispreferred_input_ids).logits

        policy_preferred_logps = policy_preferred_logits[:, :-1, :].log_softmax(dim=-1).gather(2, preferred_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        policy_dispreferred_logps = policy_dispreferred_logits[:, :-1, :].log_softmax(dim=-1).gather(2, dispreferred_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        reference_preferred_logps = reference_preferred_logits[:, :-1, :].log_softmax(dim=-1).gather(2, preferred_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        reference_dispreferred_logps = reference_dispreferred_logits[:, :-1, :].log_softmax(dim=-1).gather(2, dispreferred_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        loss = preference_loss(policy_preferred_logps, policy_dispreferred_logps, reference_preferred_logps, reference_dispreferred_logps, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_policy_preferred_loss += policy_preferred_logps.mean().item()
        total_policy_dispreferred_loss += policy_dispreferred_logps.mean().item()
        total_reference_preferred_loss += reference_preferred_logps.mean().item()
        total_reference_dispreferred_loss += reference_dispreferred_logps.mean().item()

        wandb.log({
          "[Batch] Total Loss": loss.item(),
          "[Batch] Policy Model Loss (Preferred Responses)": policy_preferred_logps.mean().item(),
          "[Batch] Policy Model Loss (Dispreferred Responses)": policy_dispreferred_logps.mean().item(),
          "[Batch] Reference Model Loss (Preferred Responses)": reference_preferred_logps.mean().item(),
          "[Batch] Reference Model Loss (Dispreferred Responses)": reference_dispreferred_logps.mean().item()
        })

    avg_loss = total_loss / len(dataloader)
    avg_policy_preferred_loss = total_policy_preferred_loss / len(dataloader)
    avg_policy_dispreferred_loss = total_policy_dispreferred_loss / len(dataloader)
    avg_reference_preferred_loss = total_reference_preferred_loss / len(dataloader)
    avg_reference_dispreferred_loss = total_reference_dispreferred_loss / len(dataloader)

    wandb.log({
        "Overall DPO Loss": avg_loss,
        "Policy Model Loss (Preferred Responses)": avg_policy_preferred_loss,
        "Policy Model Loss (Dispreferred Responses)": avg_policy_dispreferred_loss,
        "Reference Model Loss (Preferred Responses)": avg_reference_preferred_loss,
        "Reference Model Loss (Dispreferred Responses)": avg_reference_dispreferred_loss
    })

    # Save the model after each epoch
    if dist.get_rank() == 0:
        policy_model.save_pretrained(f"./policy_model_epoch_{epoch}")

def main():
    JSONL_PATH = "./data/cai/jsonl_final.jsonl"

    model_path = "./llama3-mathlib-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = PreferenceDataset(JSONL_PATH, tokenizer)

    policy_model = AutoModelForCausalLM.from_pretrained(model_path)
    reference_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    policy_model = policy_model.to(dist.get_rank())
    reference_model = reference_model.to(dist.get_rank())
    policy_model = torch.nn.parallel.DistributedDataParallel(policy_model, device_ids=[dist.get_rank()])
    reference_model = torch.nn.parallel.DistributedDataParallel(reference_model, device_ids=[dist.get_rank()])

    # Create a DistributedSampler for the dataset
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    num_epochs = 10
    beta = 0.1

    if dist.get_rank() == 0:
        wandb.init(project="dpo_training", config={"num_epochs": num_epochs, "beta": beta})

    for epoch in range(num_epochs):
        train(policy_model, reference_model, dataloader, optimizer, beta, epoch)

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    main()